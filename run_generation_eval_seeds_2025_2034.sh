#!/bin/bash
set -euo pipefail

cd /home/yiqiuliu/DreamDiffusion
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion

GPU_ID="${GPU_ID:-2}"
SEEDS=(2025 2026 2027 2028 2029 2030 2031 2032 2033 2034)

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="/home/yiqiuliu/DreamDiffusion/outputs/generation_visual200_strength0.7_seed${SEED}"
  LOG_PATH="${OUT_DIR}/run.log"
  mkdir -p "${OUT_DIR}"

  echo "=========================================="
  echo "Seed: ${SEED}"
  echo "Output: ${OUT_DIR}"
  echo "Log: ${LOG_PATH}"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python code/test_generation_eval.py \
    --output_dir "${OUT_DIR}" \
    --eval_split visual_test200 \
    --source_split visual_test200 \
    --candidate_strategy top1 \
    --checkpoint_path /home/yiqiuliu/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth \
    --num_items 200 \
    --ddim_steps 50 \
    --strengths 0.7 \
    --seed "${SEED}" \
    --compute_metrics true \
    2>&1 | tee "${LOG_PATH}"
done

echo "All seeds completed."
