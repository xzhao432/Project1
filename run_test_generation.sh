#!/usr/bin/env bash
set -euo pipefail

# DreamDiffusion Image Generation Test Script
# Usage: CHECKPOINT=/path/to/checkpoint.pth ./run_test_generation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion

# Python binary
PYTHON_BIN="${PYTHON_BIN:-python}"

# Data paths
DATA_DIR="${DATA_DIR:-/home/yiqiuliu/DL_Project}"
EEG_SIGNALS="${EEG_SIGNALS:-$DATA_DIR/train_for_dream.pt}"
PRECOMPUTED_TEST="${PRECOMPUTED_TEST:-$ROOT_DIR/precomputed_features/test_precomputed.h5}"

# Model paths
CHECKPOINT="${CHECKPOINT:-}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/home/yiqiuliu/DreamDiffusion_old/pretrains}"

# Generation parameters
NUM_SAMPLES="${NUM_SAMPLES:-5}"
DDIM_STEPS="${DDIM_STEPS:-250}"
LIMIT="${LIMIT:-}"

# Output
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/generation_test}"
SEED="${SEED:-2025}"
SUBJECT="${SUBJECT:-0}"

# Validation
if [[ -z "$CHECKPOINT" ]]; then
  echo "[error] Please set CHECKPOINT=/path/to/checkpoint.pth" >&2
  echo "Example: CHECKPOINT=./output/checkpoint_best.pth ./run_test_generation.sh" >&2
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[error] Checkpoint file not found: $CHECKPOINT" >&2
  exit 2
fi

# Build command
CMD=(
  "$PYTHON_BIN" -u "$ROOT_DIR/evaluate_generation.py"
  --checkpoint "$CHECKPOINT"
  --data-path "$EEG_SIGNALS"
  --imagenet-path "$DATA_DIR"
  --pretrain-root "$PRETRAIN_ROOT"
  --out-dir "$OUT_DIR"
  --num-samples "$NUM_SAMPLES"
  --ddim-steps "$DDIM_STEPS"
  --seed "$SEED"
  --subject "$SUBJECT"
)

# Add optional limit parameter
if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

# Add any additional arguments passed to this script
CMD+=("$@")

# Print configuration
echo "============================================"
echo "DreamDiffusion Generation Test"
echo "============================================"
echo "Checkpoint:      $CHECKPOINT"
echo "Data directory:  $DATA_DIR"
echo "EEG signals:     $EEG_SIGNALS"
echo "Precomputed:     $PRECOMPUTED_TEST"
echo "Output:          $OUT_DIR"
echo "Num samples:     $NUM_SAMPLES"
echo "DDIM steps:      $DDIM_STEPS"
echo "Subject:         $SUBJECT"
echo "Seed:            $SEED"
if [[ -n "$LIMIT" ]]; then
  echo "Limit:           $LIMIT"
fi
echo "============================================"
echo ""

# Execute
exec "${CMD[@]}"
