#!/usr/bin/env bash
# Weight-freeze PTQ of MambaIRv2-Light (Table 1): x2 / x4, W4A4 / W2A2.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ckpts logs
for SCALE in 2 4; do
  [[ -f "calib/calib_x${SCALE}.pth" ]] || python data/make_calib_set.py --scale "${SCALE}"
  for BIT in 4 2; do
    TAG="mambairv2_light_x${SCALE}_w${BIT}a${BIT}"
    python calibrate.py --scale "${SCALE}" --bit "${BIT}" --save "ckpts/${TAG}.pth" 2>&1 | tee "logs/${TAG}.log"
    python infer.py --ckpt "ckpts/${TAG}.pth" 2>&1 | tee -a "logs/${TAG}.log"
  done
done
