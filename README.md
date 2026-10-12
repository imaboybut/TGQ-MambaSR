# TGQ-MambaSR: Hardware-Friendly Token-Group Activation Quantization for Low-Bit Mamba Super-Resolution

**NeurIPS 2026** · [Poster page](https://neurips.cc/virtual/2026/loc/paris/poster/155635)

Jinwoo Chung, Jangho Kim\* · Kookmin University (\*corresponding author)

TGQ-MambaSR is the official PyTorch implementation of **Token-Group Quantization (TGQ)**, a
post-training quantization (PTQ) method for low-bit (W4A4 / W2A2) **Mamba-based image
super-resolution** (MambaIRv2, MambaIR). Selective scan, the input-dependent state-space
operator of Mamba, spreads spatial tokens over different activation ranges, so one clipping range
per layer fits them poorly. TGQ predicts a token group from pre-scan features with a frozen
two-layer FFN and looks up offline-calibrated, group-wise clipping bounds: no runtime statistics,
no dynamic clipping, no post-scan grouping, hence static low-bit inference on edge hardware.
A **Frequency-Preserving Refinement (FPR)** objective aligns the Fourier amplitude and phase of the
quantized student with the full-precision teacher while the clipping bounds are refined.

<p align="center"><img src="assets/overview.png" width="85%" alt="Token-Group Quantization overview"></p>

**Keywords:** post-training quantization · PTQ · low-bit quantization · activation quantization ·
token-group quantization · Mamba · state space model (SSM) · selective scan · MambaIR · MambaIRv2 ·
image super-resolution (SISR) · image restoration · JPEG artifact removal · image denoising ·
SwinIR · W4A4 · W2A2 · 4-bit · 2-bit · model compression · efficient inference · edge deployment ·
Jetson Orin Nano · NeurIPS 2026

## Contents

| Path | What it is |
|---|---|
| `calibrate.py` | TGQ calibration for SR on MambaIRv2-Light / MambaIRv2-Base / SwinIR-light (`--protocol freeze` or `lora`) |
| `calibrate_ir.py` | TGQ calibration for MambaIRv2-Base on JPEG artifact removal (CAR) and color denoising |
| `infer.py`, `infer_ir.py` | Rebuild a calibrated model from its checkpoint and evaluate on the benchmarks |
| `tgq/` | The method: backbone site specs, descriptors + K-means pseudo-labels, pre-scan predictor, group-conditional quantizer swap, FPR / feature losses, calibration pipeline |
| `quant/` | Fake quantizers with learnable clipping bounds, MSE bound search, quantized Linear / Conv2d, LoRA rectifier group, quantization scopes |
| `arch/` | MambaIRv2 (with a pure-PyTorch parallel-scan fallback) and SwinIR backbones |
| `data/` | DF2K / benchmark loaders, restoration loaders, calibration-set builders |
| `scripts/` | Shell scripts that reproduce every setting of the paper |

## Installation

```bash
git clone https://github.com/imaboybut/TGQ-MambaSR.git && cd TGQ-MambaSR
pip install -r requirements.txt        # torch>=2.0, numpy, einops, Pillow, scikit-learn
```

`mamba_ssm` is optional. When it is importable, the selective scan of MambaIRv2 runs on its CUDA
kernel; otherwise a pure-PyTorch Blelloch parallel scan is used (slower and more memory-hungry,
which matters for MambaIRv2-Base). The CUDA kernel skips the post-scan hidden-state quantizer.

## Data and full-precision weights

Nothing is downloaded automatically. Place the files as follows (or point the `--*-root` flags
elsewhere).

```
pretrained/
├── mambairv2_lightSR_x{2,4}.pth                       # MambaIRv2 release
├── mambairv2_classicSR_Base_x{2,4}.pth                # MambaIRv2 release (optional, --model base)
├── mambairv2_jpeg_q{10,30,40}.pth, mambairv2_ColorDN_15.pth   # MambaIRv2 release (optional, IR tasks)
└── 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{2,4}.pth  # SwinIR release (optional, --backbone swinir)
datasets/
├── DF2K/DF2K_train_HR/*.png                           # calibration / refinement (LR only is used)
├── DF2K/DF2K_train_LR_bicubic/X{2,4}/{stem}x{scale}.png
├── benchmark/{Set5,Set14,B100,Urban100,Manga109}/{HR,LR_bicubic/X{2,4}}/
│   # LR names: {stem}x{scale}.png; Manga109 uses {stem}_LRBI_x{scale}.png
├── DIV2K/DIV2K_valid_HR/*.png                         # optional, IR calibration patches
└── benchmark_ir/CAR/{classic5,LIVE1}/..., benchmark_ir/ColorDN/{CBSD68HQ,Kodak24HQ,McMasterHQ,Urban100HQ}/
```

Full-precision weights: [MambaIR / MambaIRv2](https://github.com/csguoh/MambaIR) and
[SwinIR](https://github.com/JingyunLiang/SwinIR). Benchmarks are the standard SR / restoration
test sets used by those repositories.

## Calibrate MambaIRv2-Light (main setting)

```bash
# 1) sample the calibration set: 96 LR patches from the DF2K training split
python data/make_calib_set.py --scale 2

# 2) calibrate (W4A4; use --bit 2 for W2A2, --scale 4 for x4)
python calibrate.py --scale 2 --bit 4 --save ckpts/mambairv2_light_x2_w4a4.pth

# 3) evaluate on Set5 / Set14 / B100 / Urban100 / Manga109
python infer.py --ckpt ckpts/mambairv2_light_x2_w4a4.pth
```

`scripts/calibrate_mambairv2_light.sh` runs all four settings (x2 / x4 × W4A4 / W2A2).
Several GPUs can be used with `torchrun --standalone --nproc_per_node=N calibrate.py ...`
(`--batch-size` is then the per-process batch).

What `calibrate.py` does, in order:

1. Build the FP teacher and an FP-initialised student; inject the quantization scope
   (weights and activations of all linear / pointwise-conv layers, window-attention operands,
   prompt library and the five selective-scan operands).
2. Initialise every clipping bound by MSE search on the calibration patches.
3. Collect the per-token descriptor `[q0.99, q0.999, RMS, max]` of |x| at the eight
   group-conditional sites of each ASSM block (`assm.out_proj`, `convffn2.fc1`, `convffn2.fc2`,
   and the scan operands `xs, dts, delta, Bs, Cs`), run one K-means (K = 8) over all
   (token, block) pairs, and train the shared pre-scan predictor (C → 64 → K, GELU) with
   cross-entropy on the pseudo-labels (Adam 1e-3, batch 1024, 20 epochs, 80/20 split).
4. Replace the activation quantizers at those sites with group-conditional quantizers and
   initialise their bound tables group by group under the frozen predictor.
5. Refine only the clipping bounds against the teacher with
   `L = L_pix + 5 · L_feat + 0.01 · L_FPR` for 3,000 iterations (batch 8, Adam 1e-2, cosine
   schedule). Only LR images are used; no HR target enters the calibration.

## Other protocols and backbones

```bash
# Weight-training PTQ with LoRA rectifier groups (r = 8, N = 4), Table 3
python calibrate.py --scale 2 --bit 4 --protocol lora --save ckpts/mambairv2_light_x2_w4a4_lora.pth

# MambaIRv2-Base, x2 classic SR (Appendix B); tile the validation / inference for large images
python calibrate.py --model base --scale 2 --bit 4 --tile 128 --save ckpts/mambairv2_base_x2_w4a4.pth

# SwinIR-light transfer (Appendix B.1): group-conditional MLP sites, pre-block predictor
python calibrate.py --backbone swinir --scale 2 --bit 4 --save ckpts/swinir_light_x2_w4a4.pth
python calibrate.py --backbone swinir --scale 2 --bit 4 --protocol lora --save ckpts/swinir_light_x2_w4a4_lora.pth

# MambaIRv2-Base on JPEG artifact removal / color denoising (Appendix A)
python data/make_ir_calib_sets.py --hr-dir datasets/DIV2K/DIV2K_valid_HR
python calibrate_ir.py --task CAR-q10 --bit 4 --save ckpts/mambairv2_base_car_q10_w4a4.pth
python infer_ir.py --ckpt ckpts/mambairv2_base_car_q10_w4a4.pth
```

`infer.py` / `infer_ir.py` read the backbone, protocol and group count from the checkpoint and
rebuild the quantized graph before loading it. See `scripts/` for the complete sweeps.

Note that the ASSM of MambaIRv2 routes tokens with a Gumbel-softmax, so its forward pass is
stochastic; `infer.py` / `infer_ir.py` fix the random seed and enable deterministic kernels so the
reported numbers are reproducible on the same hardware (SwinIR-light is deterministic).

## Results: MambaIRv2-Light, weight-freeze PTQ (PSNR / SSIM on Y)

| Scale | W/A | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ×2 | 32/32 (FP) | 38.26 / 0.9615 | 34.09 / 0.9221 | 32.36 / 0.9019 | 33.26 / 0.9378 | 39.35 / 0.9785 |
| ×2 | 4/4 | 37.56 / 0.9578 | 33.24 / 0.9149 | 31.86 / 0.8950 | 31.34 / 0.9204 | 37.85 / 0.9742 |
| ×2 | 2/2 | 35.02 / 0.9436 | 31.41 / 0.8966 | 30.51 / 0.8766 | 27.90 / 0.8708 | 32.67 / 0.9508 |
| ×4 | 32/32 (FP) | 32.51 / 0.8992 | 28.84 / 0.7878 | 27.75 / 0.7426 | 26.82 / 0.8079 | 31.24 / 0.9182 |
| ×4 | 4/4 | 31.46 / 0.8811 | 28.14 / 0.7683 | 27.24 / 0.7231 | 25.39 / 0.7600 | 29.21 / 0.8884 |
| ×4 | 2/2 | 29.12 / 0.8260 | 26.57 / 0.7259 | 26.32 / 0.6891 | 23.57 / 0.6805 | 25.58 / 0.8031 |

Image borders are cropped by the scale factor before computing the metrics. Results for the LoRA
protocol, MambaIRv1, MambaIRv2-Base, SwinIR-light, restoration tasks and the Jetson Orin Nano
measurements are in the paper.

## Using TGQ in your own PTQ pipeline

The method is backbone-agnostic. `tgq/backbones.py` declares, per backbone, the residual-block
name pattern, the module suffixes whose input activations get group-wise bounds, the scan-operand
quantizers (token axis last) and the pre-scan feature the predictor reads. Adding a backbone means
adding one `BackboneSpec` and a quantization scope in `quant/`; `tgq/pipeline.py` then provides
`init_bounds`, `fit_token_groups`, `distill` and `restore_token_groups` unchanged.

## Citation

The BibTeX entry will be added once the proceedings entry is available. Until then please cite the
NeurIPS 2026 paper *Hardware-Friendly Token-Group Activation Quantization for Low-Bit Mamba
Super-Resolution* (Jinwoo Chung, Jangho Kim) and link to the
[poster page](https://neurips.cc/virtual/2026/loc/paris/poster/155635).

## License and acknowledgements

Released under the Apache License 2.0 (see `LICENSE`). The backbones come from
[MambaIR](https://github.com/csguoh/MambaIR) and [SwinIR](https://github.com/JingyunLiang/SwinIR);
the fake-quantization scaffolding follows the SR PTQ codebases
[2DQuant](https://github.com/Kai-Liu001/2DQuant) and SPR²Q. We thank the authors for releasing
their code.
