"""Benchmark inference for a calibrated MambaIRv2-Base restoration checkpoint.
CAR: Classic5 / LIVE1 (grayscale PSNR / SSIM). ColorDN: CBSD68 / Kodak24 / McMaster / Urban100.

  python infer_ir.py --ckpt ckpts/mambairv2_base_car_q10_w4a4.pth
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from data.ir_dataset import IRPairedEvalDataset  # noqa: E402
from eval_psnr import evaluate_ir  # noqa: E402
from models import IR_BENCHMARKS, IR_TASKS, build_ir_model  # noqa: E402
from quant.scope_mambairv2 import build_quantized_network  # noqa: E402
from tgq.backbones import MAMBAIRV2  # noqa: E402
from tgq.pipeline import load_fp_weights, mark_calibrated, restore_token_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fp-ckpt", default=None)
    ap.add_argument("--bench-root", default="datasets/benchmark_ir")
    ap.add_argument("--tile", type=int, default=0, help="tile size (0 = whole image)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(0)

    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    task, cfg = blob["task"], IR_TASKS[blob["task"]]
    net = build_ir_model(task).to(args.device)
    load_fp_weights(net, args.fp_ckpt or str(Path("pretrained") / cfg["fp_ckpt"]))
    build_quantized_network(net, bit=blob["bit"])
    net.to(args.device)
    restore_token_groups(net, blob, MAMBAIRV2, args.device)
    mark_calibrated(net)
    net.load_state_dict(blob["params"], strict=True)
    net.to(args.device).eval()
    print(f"== mambairv2-base {task} W{blob['bit']}A{blob['bit']} K={blob.get('num_groups')} ==")

    kind = "CAR" if task.startswith("CAR") else "ColorDN"
    psnrs = []
    for entry in IR_BENCHMARKS[kind]:
        if kind == "CAR":
            name, gt, lq = entry
            ds = IRPairedEvalDataset(f"{args.bench_root}/{gt}",
                                     lq_root=f"{args.bench_root}/{lq.format(q=task.split('-q')[1])}",
                                     in_chans=cfg["in_chans"])
        else:
            name, gt = entry
            ds = IRPairedEvalDataset(f"{args.bench_root}/{gt}", sigma=cfg["sigma"], in_chans=cfg["in_chans"])
        r = evaluate_ir(net, ds, args.device, ssim=True, tile=args.tile)
        print(f"  [{name:<9s}]  PSNR={r['psnr']:.4f}  SSIM={r['ssim']:.4f}  ({len(ds)} images)")
        psnrs.append(r["psnr"])
    print(f"  average   PSNR={sum(psnrs) / len(psnrs):.4f}")


if __name__ == "__main__":
    main()
