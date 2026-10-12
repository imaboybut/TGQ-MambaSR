"""PSNR / SSIM evaluation.

SR benchmarks: Y-channel PSNR / SSIM after cropping ``scale`` pixels from the
border; inputs are mirror-padded to a multiple of the window size.
IR benchmarks (CAR / ColorDN): PSNR / SSIM on the model's native channels at
scale 1, with optional tiled inference for large images.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def rgb_to_y(img01: np.ndarray) -> np.ndarray:
    """RGB in [0,1] -> Y in [16, 235]."""
    r, g, b = img01[..., 0], img01[..., 1], img01[..., 2]
    return 65.481 * r + 128.553 * g + 24.966 * b + 16.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)


def _gaussian_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 'valid' correlation with the 1-D kernel ``k`` (same values as
    cv2.filter2D with the outer-product window followed by a (len(k)//2) crop)."""
    r = len(k)
    H, W = img.shape
    tmp = sum(k[i] * img[:, i:W - r + 1 + i] for i in range(r))
    return sum(k[i] * tmp[i:H - r + 1 + i, :] for i in range(r))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on a single channel in [0,255] with an 11x11 Gaussian window (sigma 1.5)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x = np.arange(11, dtype=np.float64) - 5
    k = np.exp(-(x ** 2) / (2 * 1.5 ** 2))
    k = (k / k.sum()).astype(a.dtype)
    mu1, mu2 = _gaussian_valid(a, k), _gaussian_valid(b, k)
    s1 = _gaussian_valid(a ** 2, k) - mu1 ** 2
    s2 = _gaussian_valid(b ** 2, k) - mu2 ** 2
    s12 = _gaussian_valid(a * b, k) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return float(ssim_map.mean())


def calculate_psnr_y(sr: np.ndarray, hr: np.ndarray, crop: int) -> float:
    if crop > 0:
        sr, hr = sr[crop:-crop, crop:-crop], hr[crop:-crop, crop:-crop]
    return _psnr(rgb_to_y(sr / 255.0), rgb_to_y(hr / 255.0))


def calculate_ssim_y(sr: np.ndarray, hr: np.ndarray, crop: int) -> float:
    if crop > 0:
        sr, hr = sr[crop:-crop, crop:-crop], hr[crop:-crop, crop:-crop]
    return _ssim(rgb_to_y(sr / 255.0), rgb_to_y(hr / 255.0))


@torch.no_grad()
def inference_padded(model: nn.Module, lq: torch.Tensor, window_size: int = 16,
                     scale: int = 2, tile: int = 0) -> torch.Tensor:
    """Mirror-pad to a multiple of ``window_size``; ``tile > 0`` splits large LR
    images into overlapping tiles (overlap = window_size) and averages the seams."""

    def _forward(x):
        _, _, h, w = x.shape
        mh, mw = (-h) % window_size, (-w) % window_size
        x = torch.cat([x, torch.flip(x, dims=[2])], dim=2)[:, :, : h + mh, :]
        x = torch.cat([x, torch.flip(x, dims=[3])], dim=3)[:, :, :, : w + mw]
        return model(x)[:, :, : h * scale, : w * scale]

    _, _, h, w = lq.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return _forward(lq)
    stride = max(1, tile - window_size)
    ys = list(range(0, max(1, h - tile + 1), stride)) + [max(0, h - tile)]
    xs = list(range(0, max(1, w - tile + 1), stride)) + [max(0, w - tile)]
    out = torch.zeros(lq.size(0), lq.size(1), h * scale, w * scale, device=lq.device, dtype=lq.dtype)
    cnt = torch.zeros_like(out)
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += _forward(lq[:, :, y0:y1, x0:x1])
            cnt[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] += 1
    return out / cnt.clamp_min(1)


@torch.no_grad()
def evaluate_on_dataset(model: nn.Module, dataset, device: str = "cuda", scale: int = 2,
                        window_size: int = 16, ssim: bool = True, tile: int = 0) -> Dict[str, float]:
    model.eval()
    psnrs, ssims = [], []
    for item in dataset:
        gt = item["gt_np"]
        sr = inference_padded(model, item["lq"].unsqueeze(0).to(device), window_size, scale, tile)
        sr = sr.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        sr = sr[: gt.shape[0], : gt.shape[1]]
        psnrs.append(calculate_psnr_y(sr, gt, crop=scale))
        if ssim:
            ssims.append(calculate_ssim_y(sr, gt, crop=scale))
    out = {"psnr": float(np.mean(psnrs))}
    if ssim:
        out["ssim"] = float(np.mean(ssims))
    return out


@torch.no_grad()
def evaluate_ir(model: nn.Module, dataset, device: str = "cuda", ssim: bool = True,
                window_size: int = 16, tile: int = 0) -> Dict[str, float]:
    """Scale-1 restoration: PSNR / SSIM over the native channels (grayscale for CAR,
    RGB averaged over channels for ColorDN)."""
    model.eval()
    psnrs, ssims = [], []
    for item in dataset:
        gt = item["gt_np"]
        out = inference_padded(model, item["lq"].unsqueeze(0).to(device), window_size, scale=1, tile=tile)
        sr = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        sr = sr[: gt.shape[0], : gt.shape[1]]
        psnrs.append(_psnr(sr, gt))
        if ssim:
            ssims.append(float(np.mean([_ssim(sr[..., c], gt[..., c]) for c in range(gt.shape[-1])])))
    res = {"psnr": float(np.mean(psnrs))}
    if ssim:
        res["ssim"] = float(np.mean(ssims))
    return res
