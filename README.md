# VisualEEGDecoding Evaluation

SwanLab evaluation logs are available at: https://swanlab.cn/@yliu886/DSAA2012/runs/jd045sna9nsv73tjp5a4f/chart

## Evaluation

```bash
python test_retrieval.py \
  --output_dir /VisualEEGDecoding/outputs/evaluation_seed2025 \
  --test_pt_path /VisualEEGDecoding/data/test.pt \
  --image_root /VisualEEGDecoding/data/images \
  --checkpoint_path /VisualEEGDecoding/runs/ablation_retrieval/channels-all_wd-0.0_temp-0.07_seed-2027/best.pth \
  --seed 2025 \
  --compute_metrics true
```
