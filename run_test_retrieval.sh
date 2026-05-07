#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-/home/yiqiuliu/DL_Project/image-eeg-data}"
CLIP_FEATURES="${CLIP_FEATURES:-$ROOT_DIR/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt}"
CHECKPOINT="${CHECKPOINT:-}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TOP_K="${TOP_K:-5}"
SEED="${SEED:-2025}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/retrieval_eval}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "[error] Please set CHECKPOINT=/path/to/best.pth or pass --checkpoint explicitly." >&2
  echo "Example: CHECKPOINT=$ROOT_DIR/runs/retrieval_xxx/best.pth bash $ROOT_DIR/run_test_retrieval.sh" >&2
  exit 2
fi

exec "$PYTHON_BIN" -u "$ROOT_DIR/test_retrieval.py" \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --clip-features "$CLIP_FEATURES" \
  --out-dir "$OUT_DIR" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --top-k "$TOP_K" \
  --seed "$SEED" \
  "$@"
