#!/usr/bin/env bash
set -euo pipefail

# DreamDiffusion Evaluation Script (with real ground truth images)
# Usage: CHECKPOINT=/path/to/checkpoint.pth ./run_evaluation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion

# Data paths
DATA_PATH="${DATA_PATH:-/home/yiqiuliu/DL_Project/train_for_dream.pt}"
IMAGENET_PATH="${IMAGENET_PATH:-/home/yiqiuliu/DL_Project/image-eeg-data}"

# Model paths
CHECKPOINT="${CHECKPOINT:-}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/home/yiqiuliu/DreamDiffusion_old/pretrains}"

# Generation parameters
NUM_SAMPLES="${NUM_SAMPLES:-5}"
DDIM_STEPS="${DDIM_STEPS:-250}"
LIMIT="${LIMIT:-}"

# Output
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/evaluation}"
SEED="${SEED:-2025}"
SUBJECT="${SUBJECT:-0}"

# Validation
if [[ -z "$CHECKPOINT" ]]; then
  echo "[error] Please set CHECKPOINT=/path/to/checkpoint.pth" >&2
  echo "Example: CHECKPOINT=./output/checkpoint_best.pth ./run_evaluation.sh" >&2
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[error] Checkpoint file not found: $CHECKPOINT" >&2
  exit 2
fi

# Build command
CMD=(
  python -u "$ROOT_DIR/evaluate_generation.py"
  --checkpoint "$CHECKPOINT"
  --data-path "$DATA_PATH"
  --imagenet-path "$IMAGENET_PATH"
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
echo "DreamDiffusion Evaluation (Real GT Images)"
echo "============================================"
echo "Checkpoint:      $CHECKPOINT"
echo "Data path:       $DATA_PATH"
echo "ImageNet path:   $IMAGENET_PATH"
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
