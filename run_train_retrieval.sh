#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Optional fast path: only build paper-style RN50 multi-blur features.
if [[ "${1:-}" == "--make-rn50-features-only" ]]; then
  exec conda run --no-capture-output -n "${FEATURE_ENV:-py311}" \
    python "$ROOT_DIR/preprocess/make_multiblur_rn50_features.py" \
    --data-root "${THINGS_EEG_ROOT:-$ROOT_DIR/data/things-eeg}" \
    --backend "${RN50_BACKEND:-open_clip}" \
    --clip-weights "${RN50_WEIGHTS:-/home/yiqiuliu/VisualEEGDecoding/data/open_clip_pytorch_model.bin}" \
    --batch-size "${RN50_BATCH_SIZE:-128}"
fi

GPU_IDS="${GPU_IDS:-6}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-/home/yiqiuliu/DL_Project/image-eeg-data}"
CLIP_FEATURES="${CLIP_FEATURES:-$ROOT_DIR/data/things-eeg/Image_feature/MultiBlur_RN50_train.pt}"
RUN_ID="${RUN_ID:-$(date +%d-%m-%Y-%H-%M-%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/retrieval_${RUN_ID}}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
SEED="${SEED:-2025}"
VAL_RATIO="${VAL_RATIO:-0.15}"
USE_TEST_AS_VAL="${USE_TEST_AS_VAL:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"

# Paper-aligned RN50 feature generation (optional but recommended).
ENABLE_RN50_FEATURES="${ENABLE_RN50_FEATURES:-0}"
FEATURE_ENV="${FEATURE_ENV:-py311}"
THINGS_EEG_ROOT="${THINGS_EEG_ROOT:-$ROOT_DIR/data/things-eeg}"
RN50_BACKEND="${RN50_BACKEND:-open_clip}"
RN50_WEIGHTS="${RN50_WEIGHTS:-/home/yiqiuliu/VisualEEGDecoding/data/open_clip_pytorch_model.bin}"
RN50_BATCH_SIZE="${RN50_BATCH_SIZE:-128}"

mkdir -p "$OUT_DIR"
LOG_FILE="${LOG_FILE:-$OUT_DIR/train.log}"

echo "[info] ROOT_DIR=$ROOT_DIR"
echo "[info] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[info] DATA_DIR=$DATA_DIR"
echo "[info] CLIP_FEATURES=$CLIP_FEATURES"
echo "[info] OUT_DIR=$OUT_DIR"
echo "[info] LOG_FILE=$LOG_FILE"
echo "[info] ENABLE_RN50_FEATURES=$ENABLE_RN50_FEATURES"

EXTRA_ARGS=()
if [[ "$USE_TEST_AS_VAL" == "1" ]]; then
  EXTRA_ARGS+=(--use-test-as-val)
fi

if command -v stdbuf >/dev/null 2>&1; then
  exec > >(stdbuf -oL -eL tee -a "$LOG_FILE") 2>&1
else
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

if [[ "$ENABLE_RN50_FEATURES" == "1" ]]; then
  echo "[info] Building RN50 multi-blur features with env=$FEATURE_ENV ..."
  conda run --no-capture-output -n "$FEATURE_ENV" \
    python "$ROOT_DIR/preprocess/make_multiblur_rn50_features.py" \
    --data-root "$THINGS_EEG_ROOT" \
    --backend "$RN50_BACKEND" \
    --clip-weights "$RN50_WEIGHTS" \
    --batch-size "$RN50_BATCH_SIZE"
fi

exec "$PYTHON_BIN" -u "$ROOT_DIR/train_retrieval.py" \
  --data-dir "$DATA_DIR" \
  --clip-features "$CLIP_FEATURES" \
  --out-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --seed "$SEED" \
  --val-ratio "$VAL_RATIO" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --weight-decay "$WEIGHT_DECAY" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  "${EXTRA_ARGS[@]}" \
  "$@"
