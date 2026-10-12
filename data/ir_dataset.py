"""Paired datasets for JPEG artifact removal (CAR, grayscale) and color Gaussian
denoising (ColorDN). Both are scale-1 restoration tasks.

CAR:     paired (HQ PNG, LQ JPEG) patches on disk.
ColorDN: HQ images only; the noisy input is synthesised with sigma on the fly.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read(path: Path, in_chans: int, scale01: bool) -> np.ndarray:
    img = Image.open(path).convert("L" if in_chans == 1 else "RGB")
    arr = np.asarray(img, dtype=np.float32)
    if in_chans == 1:
        arr = arr[..., None]
    return arr / 255.0 if scale01 else arr


def _list_images(root: Path):
    paths = sorted(root.glob("*.png"))
    if not paths:
        paths = sorted(root.glob("*.bmp")) + sorted(root.glob("*.jpg"))
    assert paths, f"no images in {root}"
    return paths


def _augment(*arrays, use_hflip: bool, use_rot: bool):
    out = list(arrays)
    if use_hflip and random.random() < 0.5:
        out = [np.ascontiguousarray(a[:, ::-1]) for a in out]
    if use_rot:
        k = random.randint(0, 3)
        if k:
            out = [np.rot90(a, k).copy() for a in out]
    return out


class CARPairedDataset(Dataset):
    def __init__(self, gt_root: str, lq_root: str, gt_size: int = 126, in_chans: int = 1,
                 use_hflip: bool = True, use_rot: bool = True):
        self.gt_root, self.lq_root = Path(gt_root), Path(lq_root)
        self.gt_size, self.in_chans = gt_size, in_chans
        self.use_hflip, self.use_rot = use_hflip, use_rot
        self.gt_paths = _list_images(self.gt_root)

    def __len__(self):
        return len(self.gt_paths)

    def _lq_path(self, gt_path: Path) -> Path:
        for ext in ("jpg", "jpeg", "JPEG"):
            cand = self.lq_root / f"{gt_path.stem}.{ext}"
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"no LQ JPEG for {gt_path.name} in {self.lq_root}")

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        gt = _read(gt_path, self.in_chans, scale01=True)
        lq = _read(self._lq_path(gt_path), self.in_chans, scale01=True)
        H, W = lq.shape[:2]
        gs = self.gt_size
        assert H >= gs and W >= gs, f"{gt_path.name} ({H}x{W}) smaller than {gs}"
        top, left = random.randint(0, H - gs), random.randint(0, W - gs)
        gt, lq = _augment(gt[top:top + gs, left:left + gs], lq[top:top + gs, left:left + gs],
                          use_hflip=self.use_hflip, use_rot=self.use_rot)
        return {"lq": torch.from_numpy(lq.transpose(2, 0, 1)).float(),
                "gt": torch.from_numpy(gt.transpose(2, 0, 1)).float()}


class ColorDNPairedDataset(Dataset):
    def __init__(self, gt_root: str, sigma: float, gt_size: int = 128,
                 use_hflip: bool = True, use_rot: bool = True):
        self.gt_root, self.sigma, self.gt_size = Path(gt_root), float(sigma), gt_size
        self.use_hflip, self.use_rot = use_hflip, use_rot
        self.gt_paths = _list_images(self.gt_root)

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        gt = _read(gt_path, 3, scale01=True)
        H, W = gt.shape[:2]
        gs = self.gt_size
        assert H >= gs and W >= gs, f"{gt_path.name} ({H}x{W}) smaller than {gs}"
        top, left = random.randint(0, H - gs), random.randint(0, W - gs)
        (gt,) = _augment(gt[top:top + gs, left:left + gs], use_hflip=self.use_hflip, use_rot=self.use_rot)
        noise = np.random.randn(*gt.shape).astype(np.float32) * (self.sigma / 255.0)
        lq = (gt + noise).clip(0.0, 1.0)
        return {"lq": torch.from_numpy(lq.transpose(2, 0, 1)).float(),
                "gt": torch.from_numpy(gt.transpose(2, 0, 1)).float()}


class IRPairedEvalDataset(Dataset):
    """Full-image loader. Pass ``lq_root`` (CAR) or ``sigma`` (ColorDN, deterministic noise).
    Yields ``{"lq": float[c,h,w] in [0,1], "gt_np": float[H,W,c] in [0,255], "name"}``."""

    def __init__(self, gt_root: str, lq_root: Optional[str] = None, sigma: Optional[float] = None,
                 in_chans: int = 3, noise_seed: int = 0):
        assert (lq_root is None) != (sigma is None), "give exactly one of lq_root / sigma"
        self.gt_root = Path(gt_root)
        self.lq_root = Path(lq_root) if lq_root else None
        self.sigma = float(sigma) if sigma is not None else None
        self.in_chans, self.noise_seed = in_chans, noise_seed
        self.gt_paths = _list_images(self.gt_root)

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        stem = gt_path.stem
        gt = _read(gt_path, self.in_chans, scale01=False)
        if self.lq_root is not None:
            lq = None
            for ext in ("jpg", "jpeg", "JPEG", "png", "bmp"):
                cand = self.lq_root / f"{stem}.{ext}"
                if cand.is_file():
                    lq = _read(cand, self.in_chans, scale01=True)
                    break
            if lq is None:
                raise FileNotFoundError(f"no LQ for {stem} in {self.lq_root}")
        else:
            rng = np.random.RandomState(self.noise_seed + idx)
            lq = (gt / 255.0 + rng.randn(*gt.shape).astype(np.float32) * (self.sigma / 255.0)).clip(0.0, 1.0)
        return {"lq": torch.from_numpy(lq.transpose(2, 0, 1)).float(), "gt_np": gt, "name": stem}
