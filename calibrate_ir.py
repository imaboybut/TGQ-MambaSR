"""Token-Group Quantization calibration of MambaIRv2-Base for JPEG artifact
removal (CAR, grayscale) and color Gaussian denoising (ColorDN). Same pipeline
as calibrate.py with the restoration data loaders and PSNR on the native channels.

  python calibrate_ir.py --task CAR-q10 --bit 4 --save ckpts/mambairv2_base_car_q10_w4a4.pth
  torchrun --standalone --nproc_per_node=4 calibrate_ir.py --task ColorDN-l15 --bit 4 --save ...
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.ir_dataset import CARPairedDataset, ColorDNPairedDataset, IRPairedEvalDataset
from eval_psnr import evaluate_ir
from models import IR_TASKS, build_ir_model
from quant.scope_mambairv2 import build_quantized_network
from tgq.backbones import MAMBAIRV2
from tgq.pipeline import (
    barrier, distill, fit_token_groups, init_bounds, load_fp_weights, save_checkpoint,
    seed_all, setup_ddp, trainable_params,
)


def build_train_dataset(task: str, calib_root: str, gt_size: int):
    cfg = IR_TASKS[task]
    root = Path(calib_root) / cfg["calib_dir"]
    if task.startswith("CAR"):
        return CARPairedDataset(str(root / "gt"), str(root / "lq"), gt_size=gt_size, in_chans=cfg["in_chans"])
    return ColorDNPairedDataset(str(root / "gt"), sigma=cfg["sigma"], gt_size=gt_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(IR_TASKS))
    ap.add_argument("--bit", type=int, default=4)
    ap.add_argument("--fp-ckpt", default=None, help="default: pretrained/<task checkpoint>")
    ap.add_argument("--calib-root", default="datasets/calib_ir", help="from data/make_ir_calib_sets.py")
    ap.add_argument("--bench-root", default="datasets/benchmark_ir")
    ap.add_argument("--gt-size", type=int, default=64)
    ap.add_argument("--calib-num", type=int, default=32, help="patches for bound initialisation")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--total-iter", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--pixel-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=5.0)
    ap.add_argument("--fpr-weight", type=float, default=0.01)
    ap.add_argument("--num-groups", type=int, default=8)
    ap.add_argument("--predictor-hidden", type=int, default=64)
    ap.add_argument("--predictor-epochs", type=int, default=20)
    ap.add_argument("--descriptor-batches", type=int, default=2)
    ap.add_argument("--descriptor-samples", type=int, default=20_000)
    ap.add_argument("--search-steps", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save", required=True)
    ap.add_argument("--val-freq", type=int, default=500)
    ap.add_argument("--print-freq", type=int, default=50)
    ap.add_argument("--tile", type=int, default=128, help="tile size for validation (0 = whole image)")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    cfg = IR_TASKS[args.task]
    if args.fp_ckpt is None:
        args.fp_ckpt = str(Path("pretrained") / cfg["fp_ckpt"])
    rank, local_rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"
    seed_all(args.seed)
    log = print if rank == 0 else (lambda *_: None)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    log(f"== mambairv2-base {args.task} W{args.bit}A{args.bit} K={args.num_groups} "
        f"iters={args.total_iter} batch={args.batch_size}x{world_size} ==")

    net_F = build_ir_model(args.task).to(device).eval()
    load_fp_weights(net_F, args.fp_ckpt)
    for p in net_F.parameters():
        p.requires_grad_(False)
    net_Q = build_ir_model(args.task).to(device)
    load_fp_weights(net_Q, args.fp_ckpt)
    build_quantized_network(net_Q, bit=args.bit, search_steps=args.search_steps)
    net_Q.to(device)

    train_ds = build_train_dataset(args.task, args.calib_root, args.gt_size)
    calib_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    lq_calib = torch.cat([b["lq"] for _, b in zip(range(args.calib_num), calib_loader)], dim=0)[: args.calib_num]
    lq_calib = lq_calib.to(device)
    cnt = init_bounds(net_Q, lq_calib)
    log(f"[init] {cnt['calibrated']}/{cnt['total']} quantizers initialised on {lq_calib.shape[0]} patches")
    barrier(world_size)

    predictor, _ = fit_token_groups(
        net_Q, calib_loader, lq_calib, MAMBAIRV2, num_groups=args.num_groups, bit=args.bit,
        search_steps=args.search_steps, hidden_dim=args.predictor_hidden, epochs=args.predictor_epochs,
        descriptor_batches=args.descriptor_batches, max_samples=args.descriptor_samples,
        device=device, seed=args.seed, log=log)
    del lq_calib
    torch.cuda.empty_cache()
    barrier(world_size)

    trainable = trainable_params(net_Q)
    log(f"[refine] {len(trainable)} trainable tensors, {sum(p.numel() for p in trainable):,} values")
    sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True, seed=args.seed, drop_last=True) \
        if world_size > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_name, val_gt, val_lq = cfg["val"]
    val_ds = IRPairedEvalDataset(f"{args.bench_root}/{val_gt}",
                                 lq_root=f"{args.bench_root}/{val_lq}" if val_lq else None,
                                 sigma=cfg["sigma"], in_chans=cfg["in_chans"])
    meta = dict(backbone="mambairv2", model="base", task=args.task, bit=args.bit,
                protocol="freeze", num_groups=args.num_groups)
    best_path = args.save.replace(".pth", "_best.pth")

    best = distill(
        net_Q, net_F, MAMBAIRV2, train_dl, sampler, trainable, args.total_iter, args.lr,
        args.pixel_weight, args.feature_weight, args.fpr_weight, device, rank, local_rank, world_size,
        args.print_freq, args.val_freq,
        evaluate=lambda: evaluate_ir(net_Q, val_ds, device, ssim=False, tile=args.tile)["psnr"],
        on_best=lambda it, psnr: save_checkpoint(best_path, net_Q, predictor, iter=it, psnr=psnr, **meta),
        log=log)
    if rank == 0:
        save_checkpoint(args.save, net_Q, predictor, iter=args.total_iter, psnr=best, **meta)
        log(f"[done] saved {args.save}  (best {val_name} PSNR {best:.4f} -> {best_path})")
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
