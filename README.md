# DreamDiffusion Evaluation

Checkpoint and experiment logs are available at:

https://swanlab.cn/@yliu886/DSAA2012/runs/tr13g3gypvjpw7i9j80v6/chart

This repository contains the evaluation entrypoint used for the visual200 EEG-to-image generation experiment.


## Run Evaluation

Run the 10-seed visual200 evaluation:

```bash
GPU_ID=0 bash run_generation_eval_seeds_2025_2034.sh
```

The script evaluates seeds:

```text
2025 2026 2027 2028 2029 2030 2031 2032 2033 2034
```

The checkpoint is not tracked in this repository. Place it at:

```text
/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth
```

## Outputs

Each seed writes to:

```text
/DreamDiffusion/outputs/generation_visual200_seed<SEED>
```

Each output directory contains:

```text
generation_visual_test200_generation_summary.csv
generation_visual_test200_generation_metrics_by_sample.csv
generation_visual_test200_generation_metrics_summary.json
```

## Metrics

The evaluation records generated-image metrics:

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
  --output_dir /DreamDiffusion/outputs/generation_visual200_seed2025 \
  --eval_split visual_test200 \
  --source_split visual_test200 \
  --checkpoint_path /DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth \
  --num_items 200 \
  --ddim_steps 50 \
  --seed 2025 \
  --compute_metrics true
```
