"""Build calibration patches for JPEG artifact removal (CAR) and color denoising
(ColorDN) from the DIV2K validation HR images.

Output layout (``--out-dir``):
    CAR/q{10,30,40}/{gt,lq}/   126x126 grayscale PNG / JPEG-q pairs
    ColorDN/level15/{gt,lq}/   128x128 RGB clean / sigma-15 noisy pairs

    python data/make_ir_calib_sets.py --hr-dir datasets/DIV2K/DIV2K_valid_HR
"""

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def rand_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    assert w >= size and h >= size, f"image ({w}x{h}) smaller than crop {size}"
    x, y = random.randint(0, w - size), random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    return Image.open(buf).copy()


def add_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    return Image.fromarray(np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-dir", default="datasets/DIV2K/DIV2K_valid_HR")
    ap.add_argument("--out-dir", default="datasets/calib_ir")
    ap.add_argument("--jpeg-qualities", default="10,30,40")
    ap.add_argument("--car-patch", type=int, default=126)
    ap.add_argument("--car-per-image", type=int, default=4)
    ap.add_argument("--dn-patch", type=int, default=128)
    ap.add_argument("--dn-per-image", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    hr_paths = sorted(p for ext in ("*.png", "*.jpg", "*.bmp") for p in Path(args.hr_dir).glob(ext))
    assert hr_paths, f"no HR images in {args.hr_dir}"
    out = Path(args.out_dir)

    for q in (int(x) for x in args.jpeg_qualities.split(",")):
        gt_dir, lq_dir = out / "CAR" / f"q{q}" / "gt", out / "CAR" / f"q{q}" / "lq"
        gt_dir.mkdir(parents=True, exist_ok=True)
        lq_dir.mkdir(parents=True, exist_ok=True)
        for p in hr_paths:
            gray = Image.open(p).convert("L")
            for i in range(args.car_per_image):
                patch = rand_crop(gray, args.car_patch)
                patch.save(gt_dir / f"{p.stem}_p{i:03d}.png")
                jpeg_compress(patch, q).save(lq_dir / f"{p.stem}_p{i:03d}.jpg")
        print(f"[CAR q{q}] {len(hr_paths) * args.car_per_image} patches")

    gt_dir, lq_dir = out / "ColorDN" / f"level{int(args.sigma)}" / "gt", out / "ColorDN" / f"level{int(args.sigma)}" / "lq"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)
    for p in hr_paths:
        rgb = Image.open(p).convert("RGB")
        for i in range(args.dn_per_image):
            patch = rand_crop(rgb, args.dn_patch)
            patch.save(gt_dir / f"{p.stem}_p{i:03d}.png")
            add_noise(patch, args.sigma).save(lq_dir / f"{p.stem}_p{i:03d}.png")
    print(f"[ColorDN sigma={args.sigma:g}] {len(hr_paths) * args.dn_per_image} patches")


if __name__ == "__main__":
    main()
