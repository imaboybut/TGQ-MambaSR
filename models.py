"""Backbone configurations, default FP checkpoint names and restoration-task tables."""

from pathlib import Path

import torch.nn as nn

from tgq.backbones import MAMBAIRV2, SWINIR

MAMBAIRV2_LIGHT = dict(
    img_size=64, embed_dim=48, d_state=8, depths=[5, 5, 5, 5], num_heads=[4, 4, 4, 4],
    window_size=16, inner_rank=32, num_tokens=64, convffn_kernel_size=5, mlp_ratio=1.0,
    upsampler="pixelshuffledirect", resi_connection="1conv",
)
MAMBAIRV2_BASE = dict(
    img_size=64, in_chans=3, embed_dim=174, d_state=16, depths=[6] * 6, num_heads=[6] * 6,
    window_size=16, inner_rank=64, num_tokens=128, convffn_kernel_size=5, mlp_ratio=2.0,
    upsampler="pixelshuffle", resi_connection="1conv", mlp_type="conv",
)
MAMBAIRV2_BASE_IR = dict(   # scale-1 restoration head (JPEG CAR / color denoising)
    embed_dim=174, d_state=16, depths=[6] * 6, num_heads=[6] * 6, window_size=16,
    inner_rank=64, num_tokens=128, convffn_kernel_size=5, mlp_ratio=2.0,
    upsampler="", resi_connection="1conv", img_range=1.0, mlp_type="conv",
)
SWINIR_LIGHT = dict(
    img_size=64, window_size=8, img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60,
    num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv",
)

SR_BACKBONES = {
    "mambairv2": dict(spec=MAMBAIRV2, window_size=16),
    "swinir": dict(spec=SWINIR, window_size=8),
}

# JPEG artifact removal (grayscale) and color Gaussian denoising on MambaIRv2-Base.
IR_TASKS = {
    "CAR-q10": dict(in_chans=1, img_size=126, fp_ckpt="mambairv2_jpeg_q10.pth", calib_dir="CAR/q10",
                    sigma=None, val=("Classic5", "CAR/classic5/Classic5_HQ", "CAR/classic5/Classic5_LQ/10")),
    "CAR-q30": dict(in_chans=1, img_size=126, fp_ckpt="mambairv2_jpeg_q30.pth", calib_dir="CAR/q30",
                    sigma=None, val=("Classic5", "CAR/classic5/Classic5_HQ", "CAR/classic5/Classic5_LQ/30")),
    "CAR-q40": dict(in_chans=1, img_size=126, fp_ckpt="mambairv2_jpeg_q40.pth", calib_dir="CAR/q40",
                    sigma=None, val=("Classic5", "CAR/classic5/Classic5_HQ", "CAR/classic5/Classic5_LQ/40")),
    "ColorDN-l15": dict(in_chans=3, img_size=128, fp_ckpt="mambairv2_ColorDN_15.pth", calib_dir="ColorDN/level15",
                        sigma=15.0, val=("CBSD68", "ColorDN/CBSD68HQ", None)),
}
IR_BENCHMARKS = {
    "CAR": [("Classic5", "CAR/classic5/Classic5_HQ", "CAR/classic5/Classic5_LQ/{q}"),
            ("LIVE1", "CAR/LIVE1/LIVE1_HQ", "CAR/LIVE1/LIVE1_LQ/{q}")],
    "ColorDN": [("CBSD68", "ColorDN/CBSD68HQ"), ("Kodak24", "ColorDN/Kodak24HQ"),
                ("McMaster", "ColorDN/McMasterHQ"), ("Urban100", "ColorDN/Urban100HQ")],
}


def build_sr_model(backbone: str, model: str, scale: int) -> nn.Module:
    if backbone == "mambairv2":
        from arch.mambairv2_arch import MambaIRv2Light
        return MambaIRv2Light(upscale=scale, **(MAMBAIRV2_LIGHT if model == "light" else MAMBAIRV2_BASE))
    if backbone == "swinir":
        from arch.swinir_arch import SwinIR
        return SwinIR(upscale=scale, **SWINIR_LIGHT)
    raise ValueError(backbone)


def default_sr_fp_ckpt(backbone: str, model: str, scale: int) -> str:
    if backbone == "mambairv2":
        name = f"mambairv2_lightSR_x{scale}.pth" if model == "light" else f"mambairv2_classicSR_Base_x{scale}.pth"
    else:
        name = f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth"
    return str(Path("pretrained") / name)


def scope_module(backbone: str):
    if backbone == "mambairv2":
        import quant.scope_mambairv2 as m
    else:
        import quant.scope_swinir as m
    return m


def build_ir_model(task: str) -> nn.Module:
    from arch.mambairv2_arch import MambaIRv2Light
    cfg = IR_TASKS[task]
    return MambaIRv2Light(upscale=1, img_size=cfg["img_size"], in_chans=cfg["in_chans"], **MAMBAIRV2_BASE_IR)
