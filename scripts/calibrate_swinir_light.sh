#!/usr/bin/env bash
# SwinIR-light transfer (Appendix B.1): weight-freeze (Table 12) and LoRA (Table 13) protocols.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ckpts logs
for SCALE in 2 4; do
  [[ -f "calib/calib_x${SCALE}.pth" ]] || python data/make_calib_set.py --scale "${SCALE}"
  for BIT in 4 2; do
    for PROTO in freeze lora; do
      TAG="swinir_light_x${SCALE}_w${BIT}a${BIT}_${PROTO}"
      python calibrate.py --backbone swinir --scale "${SCALE}" --bit "${BIT}" --protocol "${PROTO}" \
          --save "ckpts/${TAG}.pth" 2>&1 | tee "logs/${TAG}.log"
      python infer.py --ckpt "ckpts/${TAG}.pth" 2>&1 | tee -a "logs/${TAG}.log"
    done
  done
done
