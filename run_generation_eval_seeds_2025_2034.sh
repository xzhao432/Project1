#!/bin/bash
set -euo pipefail

cd /home/yiqiuliu/DreamDiffusion
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion

GPU_ID="${GPU_ID:-2}"
SEEDS=(2025 2026 2027 2028 2029 2030 2031 2032 2033 2034)

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="/home/yiqiuliu/DreamDiffusion/outputs/generation_visual200_seed${SEED}"
  mkdir -p "${OUT_DIR}"

  echo "=========================================="
  echo "Seed: ${SEED}"
  echo "Output: ${OUT_DIR}"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python code/test_generation_eval.py \
    --output_dir "${OUT_DIR}" \
    --test_pt_path /home/yiqiuliu/DL_Project/image-eeg-data/test_dreamdiffusion.pt \
    --image_root /home/yiqiuliu/DL_Project/image-eeg-data \
    --checkpoint_path /home/yiqiuliu/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth \
    --seed "${SEED}" \
    --compute_metrics true
done

echo "All seeds completed."
