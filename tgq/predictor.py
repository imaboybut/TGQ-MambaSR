"""Quantization-aware pseudo-labels and the pre-scan group predictor (Sec. 5.2-5.3).

Offline calibration:
  1. ``collect_descriptors``: for every (token, block) pair, concatenate the
     4-dim descriptor [q0.99, q0.999, RMS, max] of |x| at each group-conditional
     site (Eq. 6), and record the pre-scan feature of the same token.
  2. ``kmeans_pseudo_labels``: one K-means over the pooled descriptors.
  3. ``train_predictor``: a shared two-layer FFN maps pre-scan features to the
     pseudo-labels with cross-entropy (Eq. 8).
Inference:
  ``register_predictor_hooks`` runs the frozen predictor at the start of every
  block and writes the arg-max group ids into the ``TokenGroupContext``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.quant_arch import FakeQuantizerAct, QuantConv2d, QuantLinear, TokenGroupContext
from tgq.backbones import BackboneSpec, iter_blocks


class PreScanPredictor(nn.Module):
    """Shared two-layer FFN: pre-scan token feature -> group logits."""

    def __init__(self, in_dim: int, num_groups: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_groups))

    def forward(self, x):
        return self.net(x)

    @property
    def num_groups(self) -> int:
        return self.net[2].out_features


def per_token_descriptor(x: torch.Tensor, token_dim: int) -> torch.Tensor:
    """[q0.99, q0.999, RMS, max] of |x| over every non-batch, non-token axis -> [B, L, 4]."""
    if token_dim < 0:
        token_dim = x.dim() + token_dim
    if token_dim != 1:
        x = x.movedim(token_dim, 1)
    B, L = x.shape[:2]
    a = x.reshape(B, L, -1).abs().float()
    return torch.stack([
        torch.quantile(a, 0.99, dim=-1),
        torch.quantile(a, 0.999, dim=-1),
        a.pow(2).mean(dim=-1).sqrt(),
        a.amax(dim=-1),
    ], dim=-1)


def group_sites(model: nn.Module, spec: BackboneSpec) -> Dict[str, List[Tuple[str, nn.Module, int]]]:
    """Group-conditional sites per block: ``{block: [(site_name, act_quantizer, token_dim), ...]}``."""
    sites: Dict[str, List[Tuple[str, nn.Module, int]]] = {}
    for name, m in model.named_modules():
        block = spec.block_of(name)
        if block is None:
            continue
        if any(name.endswith(s) for s in spec.linear_sites) and isinstance(m, (QuantLinear, QuantConv2d)):
            sites.setdefault(block, []).append((f"{name}.act_quantizer", m.act_quantizer, -2))
        for attr in spec.scan_sites:
            aq = getattr(m, attr, None)
            if isinstance(aq, FakeQuantizerAct):
                sites.setdefault(block, []).append((f"{name}.{attr}", aq, -1))
    return sites


def _predictor_feature_hook(module: nn.Module, spec: BackboneSpec, fn):
    """Call ``fn(feature)`` with the [B, L, C] feature the predictor consumes."""
    if spec.predictor_hook == "input":
        return module.register_forward_pre_hook(lambda _m, inp: fn(inp[0]))
    return module.register_forward_hook(lambda _m, _inp, out: fn(out))


@torch.no_grad()
def collect_descriptors(model: nn.Module, loader, spec: BackboneSpec, max_batches: int = 2,
                        max_samples: int = 100_000, device: str = "cuda",
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(descriptors [N, 4*M], features [N, C])`` over (token, block) pairs."""
    model.eval()
    sites = group_sites(model, spec)
    blocks = [(n, m) for n, m in iter_blocks(model, spec) if n in sites]
    assert blocks, "no group-conditional sites found; run build_quantized_network first"
    n_sites = {len(sites[n]) for n, _ in blocks}
    assert len(n_sites) == 1, f"blocks expose different site counts: {n_sites}"

    stats: Dict[str, List[Optional[torch.Tensor]]] = {}
    feats: Dict[str, Optional[torch.Tensor]] = {}
    handles = []
    for bname, bmod in blocks:
        for i, (_, aq, td) in enumerate(sites[bname]):
            def _h(_m, inp, _out, b=bname, i=i, td=td):
                stats[b][i] = per_token_descriptor(inp[0], td)
            handles.append(aq.register_forward_hook(_h))

        def _f(x, b=bname):
            if torch.is_tensor(x) and x.dim() == 3:
                feats[b] = x.detach().float()
        handles.append(_predictor_feature_hook(bmod.get_submodule(spec.predictor_module), spec, _f))

    desc_out, feat_out, n_total = [], [], 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches or n_total >= max_samples:
            break
        for bname, _ in blocks:
            stats[bname] = [None] * len(sites[bname])
            feats[bname] = None
        model(batch["lq"].to(device))
        for bname, _ in blocks:
            if feats[bname] is None or any(s is None for s in stats[bname]):
                continue
            ss = torch.stack(stats[bname], dim=0)                      # [M, B, L, 4]
            M, B, L, _ = ss.shape
            desc_out.append(ss.permute(1, 2, 0, 3).reshape(B * L, M * 4).cpu())
            feat_out.append(feats[bname].reshape(B * L, -1).cpu())
            n_total += B * L
    for h in handles:
        h.remove()

    descriptors = torch.cat(desc_out, dim=0)
    features = torch.cat(feat_out, dim=0)
    if descriptors.shape[0] > max_samples:
        keep = torch.randperm(descriptors.shape[0])[:max_samples]
        descriptors, features = descriptors[keep], features[keep]
    return descriptors, features


def kmeans_pseudo_labels(descriptors: torch.Tensor, num_groups: int, seed: int = 0) -> torch.Tensor:
    from sklearn.cluster import KMeans
    X = np.nan_to_num(descriptors.float().numpy())
    km = KMeans(n_clusters=num_groups, n_init=10, random_state=seed)
    return torch.from_numpy(km.fit_predict(X)).long()


def train_predictor(features: torch.Tensor, labels: torch.Tensor, num_groups: int,
                    hidden_dim: int = 64, epochs: int = 20, batch_size: int = 1024,
                    lr: float = 1e-3, device: str = "cuda", seed: int = 0, log=print) -> PreScanPredictor:
    """Cross-entropy training on an 80/20 split; returns the frozen predictor."""
    torch.manual_seed(seed)
    predictor = PreScanPredictor(features.shape[1], num_groups, hidden_dim).to(device)
    opt = torch.optim.Adam(predictor.parameters(), lr=lr)

    N = features.shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, N // 5)
    va, tr = perm[:n_val], perm[n_val:]
    X_va, y_va = features[va].float().to(device), labels[va].long().to(device)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features[tr].float(), labels[tr].long()),
        batch_size=batch_size, shuffle=True)

    log(f"[predictor] N_train={len(tr)} N_val={n_val} in_dim={features.shape[1]} "
        f"hidden={hidden_dim} K={num_groups}")
    for ep in range(epochs):
        predictor.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(predictor(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
        predictor.eval()
        with torch.no_grad():
            logits = predictor(X_va)
            va_ce = F.cross_entropy(logits, y_va).item()
            va_top1 = (logits.argmax(-1) == y_va).float().mean().item()
        log(f"[predictor] ep {ep + 1}/{epochs}  train_ce={tr_loss / max(1, tr_n):.4f}  "
            f"val_ce={va_ce:.4f}  val_top1={va_top1 * 100:.2f}%")
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)
    return predictor


def register_predictor_hooks(model: nn.Module, predictor: PreScanPredictor, ctx: TokenGroupContext,
                             spec: BackboneSpec, device: str = "cuda"):
    """Run the frozen predictor at every block entry and publish group ids to ``ctx``."""
    predictor.eval().to(device)

    def _assign(x):
        if not torch.is_tensor(x) or x.dim() != 3:
            return
        B, L, C = x.shape
        with torch.no_grad():
            ids = predictor(x.detach().reshape(-1, C).float().to(device)).argmax(dim=-1)
        ctx.set_group_ids(ids.reshape(B, L))

    return [_predictor_feature_hook(blk.get_submodule(spec.predictor_module), spec, _assign)
            for _, blk in iter_blocks(model, spec)]
