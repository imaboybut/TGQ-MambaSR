"""Teacher-supervised objectives for clipping-bound refinement (Eq. 11-13)."""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgq.backbones import BackboneSpec, iter_blocks


class FeatureBuffer:
    """Collects the residual-block outputs of the FP teacher and the quantized student."""

    def __init__(self, net_F: nn.Module, net_Q: nn.Module, spec: BackboneSpec):
        self.feats_F: List[torch.Tensor] = []
        self.feats_Q: List[torch.Tensor] = []
        self.enabled = True   # set False around validation so block outputs are not retained
        self.handles = [m.register_forward_hook(self._hook(self.feats_F)) for _, m in iter_blocks(net_F, spec)]
        self.handles += [m.register_forward_hook(self._hook(self.feats_Q)) for _, m in iter_blocks(net_Q, spec)]

    def _hook(self, store: List[torch.Tensor]):
        def _h(_m, _i, out):
            if self.enabled:
                store.append(out)
        return _h

    def clear(self):
        self.feats_F.clear()
        self.feats_Q.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def feature_loss(feats_Q: List[torch.Tensor], feats_F: List[torch.Tensor]) -> torch.Tensor:
    """Sum over blocks of MSE between L2-normalised student / teacher features."""
    total = feats_Q[0].new_zeros(())
    for fq, ff in zip(feats_Q, feats_F):
        nq = fq.norm(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-8)
        nf = ff.norm(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-8)
        total = total + F.mse_loss(fq / nq, ff / nf)
    return total


def fpr_loss(out_Q: torch.Tensor, out_F: torch.Tensor) -> torch.Tensor:
    """Frequency-Preserving Refinement (Eq. 12): Hann-windowed 2-D FFT, then
    0.5 * L1(amplitude) + 0.5 * L1(wrapped phase)."""
    H, W = out_Q.shape[-2:]
    window = (torch.hann_window(H, device=out_Q.device, dtype=out_Q.dtype).unsqueeze(-1)
              * torch.hann_window(W, device=out_Q.device, dtype=out_Q.dtype).unsqueeze(0))
    Fq = torch.fft.rfft2(out_Q * window, dim=(-2, -1))
    Ff = torch.fft.rfft2(out_F * window, dim=(-2, -1))
    amp = (Fq.abs() - Ff.abs()).abs().mean()
    phase = torch.angle(Fq) - torch.angle(Ff)
    phase = torch.remainder(phase + math.pi, 2 * math.pi) - math.pi
    return 0.5 * amp + 0.5 * phase.abs().mean()
