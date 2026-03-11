#!/bin/bash
# Training script for VSR (Video-Only) with V-JEPA 2

python train.py \
    --exp-dir ./experiments \
    --root-dir /home/rishabhjain/Desktop/Datasets/lrs2_rf/ \
    --project-wandb llama-avsr \
    --exp-name VSR_vjepa2_vitl \
    --modality video \
    --video-encoder-name vjepa2 \
    --vjepa2-model-name facebook/vjepa2-vitl-fpc64-256 \
    --llm-model meta-llama/Meta-Llama-3.1-8B \
    --add_PETF_LLM lora \
    --unfrozen_modules peft_llm \
    --reduction_lora 64 \
    --alpha 8 \
    --downsample-ratio-video 3 \
    --num-nodes 1 \
    --gpus 1 \
    --max-frames-video 1000 \
    --lr 5e-4 \
    --max-epochs 10 \
    --train-file lrs2_train_transcript_lengths_seg16s.csv \
    --val-file lrs2_val_transcript_lengths_seg16s.csv \
    --test-file lrs2_test_transcript_lengths_seg16s.csv
