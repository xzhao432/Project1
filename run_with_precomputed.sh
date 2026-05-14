#!/bin/bash

# DreamDiffusion Training with Precomputed Features
# This script uses precomputed VAE latents and CLIP features for faster training

# Configuration
SUBJECT=0
BATCH_SIZE=8
NUM_EPOCHS=200
PRECISION=32
LR=5.3e-5
ACCUMULATE_GRAD=1

# Paths
EEG_SIGNALS_PATH="/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion_512.pt"
PRETRAIN_PATH="/home/yiqiuliu/DreamDiffusion_old/pretrains"
CHECKPOINT_PATH="/home/yiqiuliu/DreamDiffusion_old/pretrains/models/eeg_pretrain_scp/checkpoint.pth"
PRECOMPUTED_TRAIN_PATH="/home/yiqiuliu/DreamDiffusion/precomputed_features/train_precomputed.h5"
PRECOMPUTED_TEST_PATH="/home/yiqiuliu/DreamDiffusion/precomputed_features/test_precomputed.h5"

# Activate conda environment
source /data/yiqiuliu/miniforge3/etc/profile.d/conda.sh
conda activate dreamdiffusion

# Run training with precomputed features
cd /home/yiqiuliu/DreamDiffusion/code

echo "=========================================="
echo "Training with PRECOMPUTED features"
echo "This skips VAE encoding and CLIP extraction"
echo "Expected speedup: 4-6x faster per epoch"
echo "=========================================="

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python eeg_ldm_precomputed.py \
    --seed 2022 \
    --dataset EEG \
    --eeg_signals_path ${EEG_SIGNALS_PATH} \
    --precomputed_train_path ${PRECOMPUTED_TRAIN_PATH} \
    --precomputed_test_path ${PRECOMPUTED_TEST_PATH} \
    --subject ${SUBJECT} \
    --pretrain_mbm_path ${CHECKPOINT_PATH} \
    --pretrain_gm_path ${PRETRAIN_PATH} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_epoch ${NUM_EPOCHS} \
    --precision ${PRECISION} \
    --accumulate_grad ${ACCUMULATE_GRAD} \
    --crop_ratio 0.2 \
    --global_pool False \
    --use_time_cond True \
    --num_samples 5 \
    --ddim_steps 250 \
    --eval_avg True

echo "Training completed!"
