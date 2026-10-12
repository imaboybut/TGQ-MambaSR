"""Paired SR datasets: DF2K training crops (calibration / refinement) and
full-image benchmark loading (Set5 / Set14 / B100 / Urban100 / Manga109)."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DF2KPairedDataset(Dataset):
    """Random ``gt_size`` crops (LR crop = gt_size / scale) with optional flip / rotation.
    LR files are ``{stem}x{scale}.png`` under ``lq_root``."""

    def __init__(self, gt_root: str, lq_root: str, scale: int, gt_size: int = 128,
                 filename_tmpl: str = "{}x{scale}", use_hflip: bool = True, use_rot: bool = True):
        self.gt_root, self.lq_root = Path(gt_root), Path(lq_root)
        self.scale, self.gt_size, self.lq_size = scale, gt_size, gt_size // scale
        self.use_hflip, self.use_rot = use_hflip, use_rot
        self.filename_tmpl = filename_tmpl.replace("{scale}", str(scale))
        self.gt_paths = sorted(self.gt_root.glob("*.png"))
        assert self.gt_paths, f"no PNGs in {gt_root}"

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx: int):
        gt_path = self.gt_paths[idx]
        lq_path = self.lq_root / f"{self.filename_tmpl.format(gt_path.stem)}.png"
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        lq = np.asarray(Image.open(lq_path).convert("RGB"), dtype=np.float32) / 255.0

        H, W, _ = lq.shape
        assert H >= self.lq_size and W >= self.lq_size, f"{lq_path.name} ({H}x{W}) smaller than {self.lq_size}"
        top, left = random.randint(0, H - self.lq_size), random.randint(0, W - self.lq_size)
        lq = lq[top:top + self.lq_size, left:left + self.lq_size]
        gt = gt[top * self.scale:top * self.scale + self.gt_size, left * self.scale:left * self.scale + self.gt_size]

        if self.use_hflip and random.random() < 0.5:
            lq, gt = np.ascontiguousarray(lq[:, ::-1]), np.ascontiguousarray(gt[:, ::-1])
        if self.use_rot:
            k = random.randint(0, 3)
            if k:
                lq, gt = np.rot90(lq, k).copy(), np.rot90(gt, k).copy()
        return {"lq": torch.from_numpy(lq.transpose(2, 0, 1)).float(),
                "gt": torch.from_numpy(gt.transpose(2, 0, 1)).float()}


class PairedEvalDataset(Dataset):
    """Full-image loader; GT is mod-cropped to a multiple of ``scale``.
    Yields ``{"lq": float[3,h,w] in [0,1], "gt_np": float[H,W,3] in [0,255], "name"}``."""

    def __init__(self, gt_root: str, lq_root: str, scale: int, filename_tmpl: str = "{}x{scale}"):
        self.gt_root, self.lq_root, self.scale = Path(gt_root), Path(lq_root), scale
        self.filename_tmpl = filename_tmpl.replace("{scale}", str(scale))
        self.gt_paths = sorted(self.gt_root.glob("*.png"))
        if not self.gt_paths:
            self.gt_paths = sorted(self.gt_root.glob("*.bmp")) + sorted(self.gt_root.glob("*.jpg"))
        assert self.gt_paths, f"no images in {gt_root}"

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        stem = gt_path.stem
        lq_path = None
        for tmpl in (self.filename_tmpl.format(stem), stem):
            for ext in ("png", "bmp", "jpg"):
                cand = self.lq_root / f"{tmpl}.{ext}"
                if cand.is_file():
                    lq_path = cand
                    break
            if lq_path is not None:
                break
        if lq_path is None:
            raise FileNotFoundError(f"no LR match for {gt_path.name} in {self.lq_root}")

        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32)
        lq = np.asarray(Image.open(lq_path).convert("RGB"), dtype=np.float32) / 255.0
        h, w = gt.shape[:2]
        gt = gt[: h - h % self.scale, : w - w % self.scale]
        return {"lq": torch.from_numpy(lq.transpose(2, 0, 1)).float(), "gt_np": gt, "name": stem}
