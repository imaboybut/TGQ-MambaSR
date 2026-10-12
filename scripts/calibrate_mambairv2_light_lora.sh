#!/usr/bin/env bash
# Weight-training PTQ with LoRA rectifiers on MambaIRv2-Light (Table 3).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ckpts logs
for SCALE in 2 4; do
  [[ -f "calib/calib_x${SCALE}.pth" ]] || python data/make_calib_set.py --scale "${SCALE}"
  for BIT in 4 2; do
    TAG="mambairv2_light_x${SCALE}_w${BIT}a${BIT}_lora"
    python calibrate.py --scale "${SCALE}" --bit "${BIT}" --protocol lora --save "ckpts/${TAG}.pth" 2>&1 | tee "logs/${TAG}.log"
    python infer.py --ckpt "ckpts/${TAG}.pth" 2>&1 | tee -a "logs/${TAG}.log"
  done
done
