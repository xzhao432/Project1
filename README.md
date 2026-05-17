# DreamDiffusion Evaluation

**Warning: this repository is not directly runnable because we did not submit the checkpoint through GitHub, SwanLab, or any other channel.**

Experiment log is available at:

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

If you have the checkpoint locally, place it at:

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
python code/test_generation_eval.py \
  --output_dir /DreamDiffusion/outputs/generation_visual200_seed2025 \
  --test_pt_path /DreamDiffusion/data/test_dreamdiffusion.pt \
  --image_root /DreamDiffusion/data/images \
  --checkpoint_path /DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-14-47-42/checkpoint_epoch6.pth \
  --seed 2025 \
  --compute_metrics true
```
