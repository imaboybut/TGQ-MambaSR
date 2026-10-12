#!/usr/bin/env bash
# MambaIRv2-Base on JPEG artifact removal (q10/q30/q40) and color denoising (sigma 15), Appendix A.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ckpts logs
[[ -d datasets/calib_ir/CAR ]] || python data/make_ir_calib_sets.py
for TASK in CAR-q10 CAR-q30 CAR-q40 ColorDN-l15; do
  for BIT in 4 2; do
    TAG="mambairv2_base_${TASK}_w${BIT}a${BIT}"
    python calibrate_ir.py --task "${TASK}" --bit "${BIT}" --save "ckpts/${TAG}.pth" 2>&1 | tee "logs/${TAG}.log"
    python infer_ir.py --ckpt "ckpts/${TAG}.pth" 2>&1 | tee -a "logs/${TAG}.log"
  done
done
