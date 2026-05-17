# DreamDiffusion Evaluation

This repository contains the evaluation entrypoint used for the visual200 EEG-to-image generation experiment.

## Environment

```bash
cd /home/yiqiuliu/DreamDiffusion
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion
```

## Run Evaluation

Run the 10-seed visual200 evaluation:

```bash
GPU_ID=2 bash run_generation_eval_seeds_2025_2034.sh
```

The script evaluates seeds:

```text
2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
```

Default checkpoint:

```text
/home/yiqiuliu/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth
```

## Outputs

Each seed writes to:

```text
/home/yiqiuliu/DreamDiffusion/outputs/generation_visual200_strength0.7_seed<SEED>
```

Each output directory contains:

```text
generation_visual_test200_generation_summary.csv
generation_visual_test200_generation_grid.png
generation_visual_test200_generation_metrics_by_sample.csv
generation_visual_test200_generation_metrics_summary.json
run.log
```

## Metrics

The evaluation records:

```text
SSIM
MSE
CLIP Similarity
Pixel Correlation
Eval SSIM
```

## Single-Seed Command

```bash
CUDA_VISIBLE_DEVICES=2 python code/test_generation_eval.py \
  --output_dir /home/yiqiuliu/DreamDiffusion/outputs/generation_visual200_strength0.7_seed2025 \
  --eval_split visual_test200 \
  --source_split visual_test200 \
  --candidate_strategy top1 \
  --checkpoint_path /home/yiqiuliu/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth \
  --num_items 200 \
  --ddim_steps 50 \
  --strengths 0.7 \
  --seed 2025 \
  --compute_metrics true
```

## Optional: Recompute Metrics From a Grid

```bash
python code/evaluate_generation_grid.py \
  --grid_path /path/to/generation_visual_test200_generation_grid.png \
  --summary_path /path/to/generation_visual_test200_generation_summary.csv \
  --strengths 0.7
```
