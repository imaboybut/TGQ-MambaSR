"""Shared calibration pipeline: bound initialisation, token-group fitting,
teacher distillation of the clipping bounds, and checkpoint (de)serialisation."""

import os
import random
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from quant.quant_arch import (
    FakeQuantizerBase, GroupedFakeQuantizerAct, QuantLinearLoRA, TokenGroupContext,
    count_fake_quantizers,
)
from tgq.backbones import BackboneSpec
from tgq.grouped import swap_to_group_quantizers
from tgq.losses import FeatureBuffer, feature_loss, fpr_loss
from tgq.predictor import (
    PreScanPredictor, collect_descriptors, kmeans_pseudo_labels, register_predictor_hooks,
    train_predictor,
)


# ---------------------------------------------------------------------------
# Process / seeding helpers
# ---------------------------------------------------------------------------
def setup_ddp():
    """Returns ``(rank, local_rank, world_size)``; initialises NCCL under ``torchrun``."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fp_weights(model: nn.Module, ckpt: str):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("params", sd.get("state_dict", sd)), strict=True)


# ---------------------------------------------------------------------------
# Bound initialisation (Eq. 10)
# ---------------------------------------------------------------------------
@torch.no_grad()
def init_bounds(model: nn.Module, lq_calib: torch.Tensor) -> dict:
    """One forward pass over the calibration patches initialises every
    uncalibrated fake quantizer by MSE bound search. Quantizers that the
    forward never reaches (the post-scan state hook ``mymodule_hs`` when the
    mamba_ssm CUDA kernel is in use) are given fixed bounds and excluded from
    refinement."""
    model.eval()
    model(lq_calib)
    for name, m in model.named_modules():
        if isinstance(m, FakeQuantizerBase) and not m.calibrated:
            assert name.endswith("mymodule_hs"), f"quantizer not reached during calibration: {name}"
            m.set_bounds(-1.0, 1.0)
            m.calibrated = True
            m.unreached = True
    return count_fake_quantizers(model)


# ---------------------------------------------------------------------------
# Token groups (Sec. 5.2-5.4)
# ---------------------------------------------------------------------------
def fit_token_groups(model: nn.Module, calib_loader, lq_calib: torch.Tensor, spec: BackboneSpec,
                     num_groups: int, bit: int, search_steps: int, hidden_dim: int, epochs: int,
                     descriptor_batches: int, max_samples: int, device: str, seed: int, log=print):
    """Pseudo-labels -> predictor -> group-conditional quantizers -> group-wise bound init.
    Returns ``(predictor, ctx)``; the predictor hooks stay registered on ``model``."""
    desc, feats = collect_descriptors(model, calib_loader, spec, max_batches=descriptor_batches,
                                      max_samples=max_samples, device=device)
    log(f"[groups] descriptors {tuple(desc.shape)}  features {tuple(feats.shape)}")
    labels = kmeans_pseudo_labels(desc, num_groups, seed=seed)
    log(f"[groups] K-means sizes: {torch.bincount(labels, minlength=num_groups).tolist()}")
    predictor = train_predictor(feats, labels, num_groups, hidden_dim=hidden_dim, epochs=epochs,
                                device=device, seed=seed, log=log)

    ctx = TokenGroupContext(num_groups)
    register_predictor_hooks(model, predictor, ctx, spec, device=device)
    n = swap_to_group_quantizers(model, ctx, spec, bit=bit, search_steps=search_steps)
    with torch.no_grad():
        model.eval()
        model(lq_calib.to(device))
    n_cal = sum(1 for m in model.modules() if isinstance(m, GroupedFakeQuantizerAct) and m.calibrated)
    assert n_cal == n, f"group-wise init reached {n_cal}/{n} grouped quantizers"
    log(f"[groups] {n} group-conditional quantizers initialised (K={num_groups})")
    return predictor, ctx


def restore_token_groups(model: nn.Module, blob: dict, spec: BackboneSpec, device: str) -> Optional[PreScanPredictor]:
    """Inference-time counterpart of ``fit_token_groups``: rebuild the grouped
    quantizers and the predictor hooks from a checkpoint blob."""
    if blob.get("predictor") is None:
        return None
    psd = blob["predictor"]
    predictor = PreScanPredictor(psd["net.0.weight"].shape[1], psd["net.2.weight"].shape[0],
                                 psd["net.0.weight"].shape[0]).to(device)
    predictor.load_state_dict(psd)
    ctx = TokenGroupContext(predictor.num_groups)
    register_predictor_hooks(model, predictor, ctx, spec, device=device)
    swap_to_group_quantizers(model, ctx, spec, bit=blob["bit"])
    return predictor


# ---------------------------------------------------------------------------
# Distillation of the clipping bounds (Sec. 5.4-5.5)
# ---------------------------------------------------------------------------
def trainable_params(model: nn.Module, lora: bool = False) -> List[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, FakeQuantizerBase) and not getattr(m, "unreached", False):
            params += [m.lower_bound, m.upper_bound]
        if lora and isinstance(m, QuantLinearLoRA):
            params += m.lora_parameters()
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            p.requires_grad_(True)
            out.append(p)
    return out


def distill(net_Q: nn.Module, net_F: nn.Module, spec: BackboneSpec, train_dl, sampler,
            trainable: List[nn.Parameter], total_iter: int, lr: float,
            pixel_weight: float, feature_weight: float, fpr_weight: float,
            device: str, rank: int, local_rank: int, world_size: int,
            print_freq: int, val_freq: int,
            evaluate: Optional[Callable[[], float]] = None,
            on_best: Optional[Callable[[int, float], None]] = None, log=print) -> float:
    """Minimise  L = L_pix + feature_weight * L_feat + fpr_weight * L_FPR  (Eq. 13)
    w.r.t. ``trainable`` with Adam + cosine annealing. Returns the best validation PSNR."""
    net_Q.eval()   # SwinIR-light has stochastic depth; MambaIRv2 has no stochastic layers
    net_F.eval()
    fb = FeatureBuffer(net_F, net_Q, spec)
    net_Q_run = DDP(net_Q, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else net_Q
    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter)

    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    it_dl = iter(train_dl)
    best = 0.0
    t0 = time.time()
    for it in range(1, total_iter + 1):
        try:
            batch = next(it_dl)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it_dl = iter(train_dl)
            batch = next(it_dl)
        lq = batch["lq"].to(device, non_blocking=True)

        fb.clear()
        out_Q = net_Q_run(lq)
        with torch.no_grad():
            out_F = net_F(lq)
        l_pix = pixel_weight * F.l1_loss(out_Q, out_F)
        l_feat = feature_weight * feature_loss(fb.feats_Q, fb.feats_F)
        loss = l_pix + l_feat
        l_fpr = fpr_weight * fpr_loss(out_Q, out_F) if fpr_weight > 0 else None
        if l_fpr is not None:
            loss = loss + l_fpr

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if rank == 0 and it % print_freq == 0:
            el = time.time() - t0
            msg = f"  iter {it:6d}/{total_iter}  l_pix={l_pix.item():.4f}  l_feat={l_feat.item():.4f}"
            if l_fpr is not None:
                msg += f"  l_fpr={l_fpr.item():.4f}"
            log(msg + f"  lr={scheduler.get_last_lr()[0]:.2e}  elapsed={el / 60:.1f}m  "
                      f"eta={el / it * (total_iter - it) / 60:.1f}m")

        if rank == 0 and evaluate is not None and (it % val_freq == 0 or it == total_iter):
            del out_Q, out_F, loss, l_pix, l_feat, l_fpr   # free the training graph before validating
            fb.clear()
            fb.enabled = False
            torch.cuda.empty_cache()
            psnr = evaluate()
            fb.enabled = True
            log(f"  [val] iter {it}  PSNR={psnr:.4f}")
            if psnr > best:
                best = psnr
                if on_best is not None:
                    on_best(it, psnr)
        barrier(world_size)

    fb.remove()
    return best


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, net: nn.Module, predictor: Optional[PreScanPredictor], **meta):
    blob = {"params": net.state_dict(),
            "predictor": predictor.state_dict() if predictor is not None else None}
    blob.update(meta)
    torch.save(blob, path)


def mark_calibrated(model: nn.Module):
    for m in model.modules():
        if isinstance(m, FakeQuantizerBase):
            m.calibrated = True
