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
CLIP_TUNE=False

# Channel Adapter Configuration
EEG_INPUT_CHANNELS=63
EEG_PRETRAINED_CHANNELS=128
ADAPTER_WARMUP_EPOCHS=2

# VisualEEGDecoding Retrieval Encoder Configuration
USE_VISUAL_EEG_ENCODER=True
RETRIEVAL_EEG_SIGNALS_PATH="/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion.pt"
VISUAL_EEG_CHECKPOINT_PATH="/home/yiqiuliu/VisualEEGDecoding/runs/ablation_retrieval/channels-all_wd-0.0_temp-0.07_seed-2027/best.pth"
VISUAL_EEG_CHANNELS=63
VISUAL_EEG_TEMPORAL_LEN=250
VISUAL_EEG_PROJ_DIM=1024
FREEZE_VISUAL_EEG_ENCODER=True
VISUAL_EEG_PROJECTOR_ONLY=True

# Paths
EEG_SIGNALS_PATH="/home/yiqiuliu/DL_Project/image-eeg-data/train_dreamdiffusion_512.pt"
PRETRAIN_PATH="/home/yiqiuliu/DreamDiffusion_old/pretrains"
CHECKPOINT_PATH="/home/yiqiuliu/DreamDiffusion_old/pretrains/models/eeg_pretrain_scp/checkpoint.pth"
RESUME_CHECKPOINT_PATH="/home/yiqiuliu/DreamDiffusion/dreamdiffusion/results/generation/15-05-2026-13-19-01/checkpoint_epoch9.pth"
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
echo "Channel Adapter: ${EEG_INPUT_CHANNELS} -> ${EEG_PRETRAINED_CHANNELS} channels"
echo "Adapter Warmup Epochs: ${ADAPTER_WARMUP_EPOCHS}"
echo "Use VisualEEG Encoder: ${USE_VISUAL_EEG_ENCODER}"
echo "Retrieval EEG Path: ${RETRIEVAL_EEG_SIGNALS_PATH}"
echo "VisualEEG Checkpoint: ${VISUAL_EEG_CHECKPOINT_PATH}"
echo "Resume Checkpoint: ${RESUME_CHECKPOINT_PATH}"
echo "VisualEEG Projector Only: ${VISUAL_EEG_PROJECTOR_ONLY}"
echo "CLIP Tune: ${CLIP_TUNE}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python eeg_ldm_precomputed.py \
    --seed 2022 \
    --dataset EEG \
    --eeg_signals_path ${EEG_SIGNALS_PATH} \
    --precomputed_train_path ${PRECOMPUTED_TRAIN_PATH} \
    --precomputed_test_path ${PRECOMPUTED_TEST_PATH} \
    --subject ${SUBJECT} \
    --pretrain_mbm_path ${CHECKPOINT_PATH} \
    --checkpoint_path ${RESUME_CHECKPOINT_PATH} \
    --pretrain_gm_path ${PRETRAIN_PATH} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_epoch ${NUM_EPOCHS} \
    --precision ${PRECISION} \
    --accumulate_grad ${ACCUMULATE_GRAD} \
    --crop_ratio 0.2 \
    --global_pool False \
    --clip_tune ${CLIP_TUNE} \
    --use_time_cond True \
    --num_samples 5 \
    --ddim_steps 250 \
    --eval_avg True \
    --eeg_input_channels ${EEG_INPUT_CHANNELS} \
    --eeg_pretrained_channels ${EEG_PRETRAINED_CHANNELS} \
    --adapter_warmup_epochs ${ADAPTER_WARMUP_EPOCHS} \
    --use_visual_eeg_encoder ${USE_VISUAL_EEG_ENCODER} \
    --retrieval_eeg_signals_path ${RETRIEVAL_EEG_SIGNALS_PATH} \
    --visual_eeg_checkpoint_path ${VISUAL_EEG_CHECKPOINT_PATH} \
    --visual_eeg_channels ${VISUAL_EEG_CHANNELS} \
    --visual_eeg_temporal_len ${VISUAL_EEG_TEMPORAL_LEN} \
    --visual_eeg_proj_dim ${VISUAL_EEG_PROJ_DIM} \
    --freeze_visual_eeg_encoder ${FREEZE_VISUAL_EEG_ENCODER} \
    --visual_eeg_projector_only ${VISUAL_EEG_PROJECTOR_ONLY}

echo "Training completed!"
