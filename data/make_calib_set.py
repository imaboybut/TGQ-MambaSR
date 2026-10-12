"""Sample the SR calibration set: ``--num`` random LR patches (with the paired
HR crop) from the DF2K training split, saved as one ``.pth`` tensor batch.
Only the LR half is used by calibration; HR is never used as a target.

    python data/make_calib_set.py --scale 2
    python data/make_calib_set.py --scale 4
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.sr_dataset import DF2KPairedDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--gt-root", default="datasets/DF2K/DF2K_train_HR")
    ap.add_argument("--lq-root", default="datasets/DF2K/DF2K_train_LR_bicubic/X{scale}")
    ap.add_argument("--gt-size", type=int, default=128)
    ap.add_argument("--num", type=int, default=96, help="number of calibration patches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="calib/calib_x{scale}.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = DF2KPairedDataset(args.gt_root, args.lq_root.format(scale=args.scale), args.scale,
                           gt_size=args.gt_size, use_hflip=True, use_rot=True)
    batch = next(iter(DataLoader(ds, batch_size=args.num, shuffle=True, num_workers=0)))
    save = Path(args.save.format(scale=args.scale))
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lq": batch["lq"], "gt": batch["gt"]}, save)
    print(f"saved {args.num} patches to {save}  lq={tuple(batch['lq'].shape)}")


if __name__ == "__main__":
    main()
