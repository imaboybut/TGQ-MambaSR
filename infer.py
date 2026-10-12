"""Benchmark inference for a calibrated SR checkpoint (MambaIRv2 / SwinIR-light).
Rebuilds the quantized graph recorded in the checkpoint and reports Y-channel
PSNR / SSIM on Set5 / Set14 / B100 / Urban100 / Manga109.

  python infer.py --ckpt ckpts/mambairv2_light_x2_w4a4.pth
"""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from data.sr_dataset import PairedEvalDataset  # noqa: E402
from eval_psnr import evaluate_on_dataset  # noqa: E402
from models import SR_BACKBONES, build_sr_model, default_sr_fp_ckpt, scope_module  # noqa: E402
from tgq.pipeline import load_fp_weights, mark_calibrated, restore_token_groups  # noqa: E402

LR_TMPL = {"Set5": "{}x{scale}", "Set14": "{}x{scale}", "B100": "{}x{scale}",
           "Urban100": "{}x{scale}", "Manga109": "{}_LRBI_x{scale}"}


def load_quantized_sr_model(ckpt: str, fp_ckpt: str = None, device: str = "cuda:0"):
    """Returns ``(model, blob)`` with the quantized graph rebuilt and weights loaded."""
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    backbone, model_name, scale = blob["backbone"], blob["model"], blob["scale"]
    net = build_sr_model(backbone, model_name, scale).to(device)
    load_fp_weights(net, fp_ckpt or default_sr_fp_ckpt(backbone, model_name, scale))
    scope = scope_module(backbone)
    scope.build_quantized_network(net, bit=blob["bit"])
    net.to(device)
    if blob.get("protocol") == "lora":
        scope.build_lora_network(net, bit=blob["bit"], r=blob["lora_r"], num_lora=blob["num_lora"])
        net.to(device)
    restore_token_groups(net, blob, SR_BACKBONES[backbone]["spec"], device)
    mark_calibrated(net)
    net.load_state_dict(blob["params"], strict=True)
    return net.to(device).eval(), blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fp-ckpt", default=None, help="FP backbone weights (default: pretrained/...)")
    ap.add_argument("--bench-root", default="datasets/benchmark")
    ap.add_argument("--sets", nargs="+", default=list(LR_TMPL))
    ap.add_argument("--tile", type=int, default=0, help="LR tile size (0 = whole image)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(0)

    net, blob = load_quantized_sr_model(args.ckpt, args.fp_ckpt, args.device)
    scale, window = blob["scale"], SR_BACKBONES[blob["backbone"]]["window_size"]
    print(f"== {blob['backbone']}-{blob['model']} x{scale} W{blob['bit']}A{blob['bit']} "
          f"protocol={blob.get('protocol')} K={blob.get('num_groups')} ==")
    psnrs, ssims = [], []
    for name in args.sets:
        ds = PairedEvalDataset(f"{args.bench_root}/{name}/HR", f"{args.bench_root}/{name}/LR_bicubic/X{scale}",
                               scale, filename_tmpl=LR_TMPL.get(name, "{}x{scale}"))
        r = evaluate_on_dataset(net, ds, args.device, scale, window, ssim=True, tile=args.tile)
        print(f"  [{name:<9s}]  PSNR(Y)={r['psnr']:.4f}  SSIM(Y)={r['ssim']:.4f}")
        psnrs.append(r["psnr"])
        ssims.append(r["ssim"])
    print(f"  average   PSNR(Y)={sum(psnrs) / len(psnrs):.4f}  SSIM(Y)={sum(ssims) / len(ssims):.4f}")


if __name__ == "__main__":
    main()
