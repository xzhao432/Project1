#!/bin/bash

# Precompute VAE latents and image features for DreamDiffusion
# This needs to be run once before training with precomputed features

# Configuration
SUBJECT=0  # 0 means use all subjects
GPU_ID=5
BATCH_SIZE=16  # Larger batch size for faster precomputation

# Paths
EEG_SIGNALS_PATH="/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion_512.pt"
SPLITS_PATH="/home/yiqiuliu/DL_Project/image-eeg-data/block_splits_random.pth"
IMAGENET_PATH="/home/yiqiuliu/DL_Project/image-eeg-data"
PRETRAIN_PATH="/data/yiqiuliu/DreamDiffusion/pretrains"
OUTPUT_PATH="/home/yiqiuliu/DreamDiffusion/precomputed_features"

# Create output directory
mkdir -p ${OUTPUT_PATH}

# Activate conda environment
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Force offline mode for transformers/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Run precomputation
cd /home/yiqiuliu/DreamDiffusion

echo "Starting feature precomputation..."
echo "This will take approximately 2-3 hours"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo ""

python precompute_features.py \
    --eeg_signals_path ${EEG_SIGNALS_PATH} \
    --splits_path ${SPLITS_PATH} \
    --imagenet_path ${IMAGENET_PATH} \
    --pretrain_path ${PRETRAIN_PATH} \
    --output_path ${OUTPUT_PATH} \
    --subject ${SUBJECT} \
    --batch_size ${BATCH_SIZE} \
    --device cuda

echo ""
echo "Precomputation completed!"
echo "You can now run training with: ./run_with_precomputed.sh"
