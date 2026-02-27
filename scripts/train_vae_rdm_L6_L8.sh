#!/usr/bin/env bash
set -euo pipefail

# Move to repo root (script is in scripts/)
cd "$(dirname "$0")/.."

python scripts/deployment/train_vae_rdm.py \
  --input results/groundstates/groundstates_L6_rdm.h5 \
  --output_dir results/vae_rdm_L6

python scripts/deployment/train_vae_rdm.py \
  --input results/groundstates/groundstates_L8_rdm.h5 \
  --output_dir results/vae_rdm_L8

