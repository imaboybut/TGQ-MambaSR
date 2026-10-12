"""Token-Group Quantization calibration for low-bit super-resolution
(MambaIRv2-Light / MambaIRv2-Base / SwinIR-light).

Pipeline
  1. FP teacher and FP-initialised student.
  2. Inject the quantization scope (fake quantizers on weights and activations).
  3. Initialise every clipping bound by MSE search on the calibration patches (Eq. 10).
  4. [--protocol lora] attach LoRA rectifier groups to every quantized linear.
  5. Collect downstream activation descriptors -> K-means pseudo-labels -> train the
     pre-scan predictor -> swap the group-conditional sites to grouped quantizers ->
     group-wise bound initialisation (Sec. 5.2-5.4).
  6. Refine the bounds (and LoRA parameters under --protocol lora) against the FP
     teacher with L_pix + 5 L_feat + 0.01 L_FPR (Eq. 13). Only LR images are used.

Single GPU
  python calibrate.py --scale 2 --bit 4 --save ckpts/mambairv2_light_x2_w4a4.pth
Multi GPU (per-process batch = --batch-size)
  torchrun --standalone --nproc_per_node=4 calibrate.py --scale 2 --bit 4 --save ...
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.sr_dataset import DF2KPairedDataset, PairedEvalDataset
from eval_psnr import evaluate_on_dataset
from models import SR_BACKBONES, build_sr_model, default_sr_fp_ckpt, scope_module
from tgq.pipeline import (
    barrier, distill, fit_token_groups, init_bounds, load_fp_weights, save_checkpoint,
    seed_all, setup_ddp, trainable_params,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mambairv2", choices=list(SR_BACKBONES))
    ap.add_argument("--model", default="light", choices=["light", "base"],
                    help="MambaIRv2 variant (ignored for swinir)")
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--bit", type=int, default=4, help="weight and activation bit-width")
    ap.add_argument("--protocol", default="freeze", choices=["freeze", "lora"],
                    help="freeze: refine clipping bounds only (Table 1); "
                         "lora: also train LoRA rectifier groups (Table 3)")
    ap.add_argument("--fp-ckpt", default=None, help="FP backbone weights (default: pretrained/...)")
    ap.add_argument("--calib", default="calib/calib_x{scale}.pth",
                    help="calibration patches from data/make_calib_set.py")
    ap.add_argument("--gt-root", default="datasets/DF2K/DF2K_train_HR")
    ap.add_argument("--lq-root", default="datasets/DF2K/DF2K_train_LR_bicubic/X{scale}")
    ap.add_argument("--bench-root", default="datasets/benchmark")
    ap.add_argument("--gt-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=None, help="default 8 (16 for swinir lora)")
    ap.add_argument("--total-iter", type=int, default=None,
                    help="default 3000 (lora: 12000 for mambairv2, 4000 for swinir)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--pixel-weight", type=float, default=1.0)
    ap.add_argument("--feature-weight", type=float, default=5.0)
    ap.add_argument("--fpr-weight", type=float, default=0.01)
    ap.add_argument("--num-groups", type=int, default=8, help="K token groups")
    ap.add_argument("--predictor-hidden", type=int, default=64)
    ap.add_argument("--predictor-epochs", type=int, default=20)
    ap.add_argument("--descriptor-batches", type=int, default=2)
    ap.add_argument("--descriptor-samples", type=int, default=100_000)
    ap.add_argument("--search-steps", type=int, default=None,
                    help="MSE bound-search grid size (default 100; 200 for mambairv2 lora)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--num-lora", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save", required=True)
    ap.add_argument("--val-set", default="Set5")
    ap.add_argument("--val-freq", type=int, default=500)
    ap.add_argument("--print-freq", type=int, default=50)
    ap.add_argument("--tile", type=int, default=0, help="LR tile size for validation (0 = whole image)")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    lora = args.protocol == "lora"
    mamba = args.backbone == "mambairv2"
    if args.batch_size is None:
        args.batch_size = 16 if (lora and not mamba) else 8
    if args.total_iter is None:
        args.total_iter = (12000 if mamba else 4000) if lora else 3000
    if args.search_steps is None:
        args.search_steps = 200 if (lora and mamba) else 100
    if args.fp_ckpt is None:
        args.fp_ckpt = default_sr_fp_ckpt(args.backbone, args.model, args.scale)
    args.calib = args.calib.format(scale=args.scale)
    args.lq_root = args.lq_root.format(scale=args.scale)
    return args


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"
    seed_all(args.seed)
    log = print if rank == 0 else (lambda *_: None)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    cfg = SR_BACKBONES[args.backbone]
    spec, window = cfg["spec"], cfg["window_size"]
    scope = scope_module(args.backbone)
    lora = args.protocol == "lora"
    log(f"== {args.backbone}-{args.model} x{args.scale} W{args.bit}A{args.bit} protocol={args.protocol} "
        f"K={args.num_groups} iters={args.total_iter} batch={args.batch_size}x{world_size} ==")

    # 1. teacher / student
    net_F = build_sr_model(args.backbone, args.model, args.scale).to(device).eval()
    load_fp_weights(net_F, args.fp_ckpt)
    for p in net_F.parameters():
        p.requires_grad_(False)
    net_Q = build_sr_model(args.backbone, args.model, args.scale).to(device)
    load_fp_weights(net_Q, args.fp_ckpt)

    # 2. quantization scope
    scope.build_quantized_network(net_Q, bit=args.bit, search_steps=args.search_steps)
    net_Q.to(device)

    # 3. layer-wise bound initialisation on the calibration patches
    lq_calib = torch.load(args.calib, map_location=device, weights_only=False)["lq"].to(device)
    cnt = init_bounds(net_Q, lq_calib)
    log(f"[init] {cnt['calibrated']}/{cnt['total']} quantizers initialised on {lq_calib.shape[0]} patches")
    barrier(world_size)

    # 4. LoRA rectifiers (weight-training protocol)
    if lora:
        n = scope.build_lora_network(net_Q, bit=args.bit, search_steps=args.search_steps,
                                     r=args.lora_r, num_lora=args.num_lora)
        net_Q.to(device)
        log(f"[lora] {n} linears wrapped (r={args.lora_r}, N={args.num_lora})")

    # 5. token groups
    calib_loader = DataLoader(
        DF2KPairedDataset(args.gt_root, args.lq_root, args.scale, args.gt_size, use_hflip=False, use_rot=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    predictor, _ = fit_token_groups(
        net_Q, calib_loader, lq_calib, spec, num_groups=args.num_groups, bit=args.bit,
        search_steps=args.search_steps, hidden_dim=args.predictor_hidden, epochs=args.predictor_epochs,
        descriptor_batches=args.descriptor_batches, max_samples=args.descriptor_samples,
        device=device, seed=args.seed, log=log)
    del lq_calib
    torch.cuda.empty_cache()
    barrier(world_size)

    # 6. bound refinement against the FP teacher
    trainable = trainable_params(net_Q, lora=lora)
    log(f"[refine] {len(trainable)} trainable tensors, {sum(p.numel() for p in trainable):,} values")
    train_ds = DF2KPairedDataset(args.gt_root, args.lq_root, args.scale, args.gt_size, use_hflip=True, use_rot=True)
    sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True, seed=args.seed, drop_last=True) \
        if world_size > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_ds = PairedEvalDataset(f"{args.bench_root}/{args.val_set}/HR",
                               f"{args.bench_root}/{args.val_set}/LR_bicubic/X{args.scale}", args.scale)
    meta = dict(backbone=args.backbone, model=args.model, scale=args.scale, bit=args.bit,
                protocol=args.protocol, num_groups=args.num_groups,
                lora_r=args.lora_r, num_lora=args.num_lora)
    best_path = args.save.replace(".pth", "_best.pth")

    best = distill(
        net_Q, net_F, spec, train_dl, sampler, trainable, args.total_iter, args.lr,
        args.pixel_weight, args.feature_weight, args.fpr_weight, device, rank, local_rank, world_size,
        args.print_freq, args.val_freq,
        evaluate=lambda: evaluate_on_dataset(net_Q, val_ds, device, args.scale, window, ssim=False,
                                             tile=args.tile)["psnr"],
        on_best=lambda it, psnr: save_checkpoint(best_path, net_Q, predictor, iter=it, psnr=psnr, **meta),
        log=log)
    if rank == 0:
        save_checkpoint(args.save, net_Q, predictor, iter=args.total_iter, psnr=best, **meta)
        log(f"[done] saved {args.save}  (best {args.val_set} PSNR {best:.4f} -> {best_path})")
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
