# Visual EEG Retrieval

This repository contains an EEG-to-image retrieval pipeline based on *Leveraging Visual Blur Perception Characteristics for EEG Decoding* (AAAI 2026). The paper maps EEG signals into a visual embedding space and evaluates decoding by retrieving the matching image from an image feature bank.

This README focuses on the evaluation scenario: running our checkpoint on a new test set.

```text
new_test_data/
  test.pt       # test EEG samples and image ids
  test_images/  # test images for those image ids
```

Evaluation checkpoint:

```text
/home/yiqiuliu/VisualEEGDecoding/runs/08-05-2026-00-45-02/best.pth
```

## Evaluation Files

```text
run_test_retrieval.sh                       # recommended evaluation entry point
test_retrieval.py                           # evaluation implementation
preprocess/make_multiblur_rn50_features.py  # generates MultiBlur_RN50_test.pt
```

## Test Data Requirement

`test.pt`, `test_images/`, and `MultiBlur_RN50_test.pt` must describe the same test images.

If the provided `test_images/` are different from the existing repository feature file, regenerate `MultiBlur_RN50_test.pt` before evaluation.

## Step 1: Prepare Test Data

Put the new test files in one directory:

```text
/path/to/new_test_data/
  test.pt
  test_images/
```

The image ids inside `test.pt` must match the filename stems inside `test_images/`. For example, an id `abc123` should correspond to an image like `abc123.jpg` or `abc123.png`.

## Step 2: Generate Test Image Features

Create a temporary workspace for the new test features:

```bash
EVAL_ROOT=/home/yiqiuliu/VisualEEGDecoding/outputs/new_test_feature_workspace
NEW_TEST_DATA=/path/to/new_test_data

mkdir -p "$EVAL_ROOT/Image_set"
mkdir -p "$EVAL_ROOT/Image_set/test_images"
cp -r "$NEW_TEST_DATA/test_images/." "$EVAL_ROOT/Image_set/test_images/"
```

Generate `MultiBlur_RN50_test.pt`:

```bash
python /home/yiqiuliu/VisualEEGDecoding/preprocess/make_multiblur_rn50_features.py \
  --data-root "$EVAL_ROOT" \
  --split test \
  --backend open_clip \
  --clip-weights /home/yiqiuliu/VisualEEGDecoding/data/open_clip_pytorch_model.bin \
  --batch-size 128
```

This writes:

```text
$EVAL_ROOT/Image_feature/MultiBlur_RN50_test.pt
```

## Step 3: Run Evaluation

Run the checkpoint on the new test set:

```bash
CHECKPOINT=/home/yiqiuliu/VisualEEGDecoding/runs/08-05-2026-00-45-02/best.pth \
DATA_DIR=/path/to/new_test_data \
CLIP_FEATURES=/home/yiqiuliu/VisualEEGDecoding/outputs/new_test_feature_workspace/Image_feature/MultiBlur_RN50_test.pt \
OUT_DIR=/home/yiqiuliu/VisualEEGDecoding/outputs/new_test_eval \
bash /home/yiqiuliu/VisualEEGDecoding/run_test_retrieval.sh
```

## Outputs

The evaluation writes:

```text
/home/yiqiuliu/VisualEEGDecoding/outputs/new_test_eval/metrics_test.json
/home/yiqiuliu/VisualEEGDecoding/outputs/new_test_eval/rankings_test.csv
/home/yiqiuliu/VisualEEGDecoding/outputs/new_test_eval/features_test.pt
```

`metrics_test.json` contains Top-1, Top-3, Top-5, mean rank, and median rank. `rankings_test.csv` contains per-sample retrieval results, including the target id, target rank, and top-k predicted ids.

## Local Example

For the existing local test set:

```bash
CHECKPOINT=/home/yiqiuliu/VisualEEGDecoding/runs/08-05-2026-00-45-02/best.pth \
DATA_DIR=/home/yiqiuliu/DL_Project/image-eeg-data \
CLIP_FEATURES=/home/yiqiuliu/VisualEEGDecoding/data/things-eeg/Image_feature/MultiBlur_RN50_test.pt \
OUT_DIR=/home/yiqiuliu/VisualEEGDecoding/outputs/local_test_eval \
bash /home/yiqiuliu/VisualEEGDecoding/run_test_retrieval.sh
```

Only use this shortcut when the local `test_images/` match the existing `MultiBlur_RN50_test.pt`.

## Citation

```bibtex
@inproceedings{liu2026leveraging,
  title={Leveraging Visual Blur Perception Characteristics for EEG Decoding},
  author={Liu, Wenchao and Li, Hongwei and Xu, Zhouyang and Ma, Lin and Li, Haifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={21},
  pages={17580--17588},
  year={2026}
}
```
