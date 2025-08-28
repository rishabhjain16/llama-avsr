# Using Auto-AVSR as Video Encoder

This document provides examples of how to use Auto-AVSR pretrained models as video encoders in Llama-AVSR.

## Prerequisites

1. Download a pretrained Auto-AVSR model from the [Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr#model-zoo)
2. Ensure you have the LRS3 dataset preprocessed according to the main README

## Training Examples

### VSR (Video-only Speech Recognition)

```bash
python train.py \
    --exp-dir ./experiments \
    --root-dir /path/to/lrs3 \
    --project-wandb my_wandb_project \
    --exp-name VSR_autoavsr_experiment \
    --modality video \
    --video-encoder-name auto-avsr \
    --pretrain-auto-avsr-path /path/to/vsr_trlrs3vox2_base.pth \
    --llm-model meta-llama/Meta-Llama-3.1-8B \
    --add_PETF_LLM lora \
    --unfrozen_modules peft_llm \
    --reduction_lora 64 \
    --alpha 8 \
    --downsample-ratio-video 3 \
    --num-nodes 1 \
    --gpus 8 \
    --max-frames-video 1000 \
    --lr 5e-4
```

### AVSR (Audio-Visual Speech Recognition)

```bash
python train.py \
    --exp-dir ./experiments \
    --root-dir /path/to/lrs3 \
    --project-wandb my_wandb_project \
    --exp-name AVSR_autoavsr_experiment \
    --modality audiovisual \
    --audio-encoder-name openai/whisper-medium.en \
    --video-encoder-name auto-avsr \
    --pretrain-auto-avsr-path /path/to/vsr_trlrs3vox2_base.pth \
    --llm-model meta-llama/Meta-Llama-3.1-8B \
    --add_PETF_LLM lora \
    --unfrozen_modules peft_llm \
    --reduction_lora 64 \
    --alpha 8 \
    --downsample-ratio-audio 3 \
    --downsample-ratio-video 3 \
    --num-nodes 1 \
    --gpus 8 \
    --max-frames-audiovisual 1000 \
    --lr 1e-3
```

## Inference Examples

### VSR Inference

```bash
python eval.py \
    --exp-name VSR_autoavsr_inference \
    --modality video \
    --project-wandb my_wandb_project \
    --pretrained-model-path /path/to/trained_model.pth \
    --root-dir /path/to/lrs3 \
    --video-encoder-name auto-avsr \
    --pretrain-auto-avsr-path /path/to/vsr_trlrs3vox2_base.pth \
    --llm-model meta-llama/Meta-Llama-3.1-8B \
    --add_PETF_LLM lora \
    --reduction_lora 64 \
    --alpha 8 \
    --downsample-ratio-video 3 \
    --max-dec-tokens 32 \
    --num-beams 15
```

### AVSR Inference

```bash
python eval.py \
    --exp-name AVSR_autoavsr_inference \
    --modality audiovisual \
    --project-wandb my_wandb_project \
    --pretrained-model-path /path/to/trained_model.pth \
    --root-dir /path/to/lrs3 \
    --audio-encoder-name openai/whisper-medium.en \
    --video-encoder-name auto-avsr \
    --pretrain-auto-avsr-path /path/to/vsr_trlrs3vox2_base.pth \
    --llm-model meta-llama/Meta-Llama-3.1-8B \
    --add_PETF_LLM lora \
    --reduction_lora 64 \
    --alpha 8 \
    --downsample-ratio-audio 3 \
    --downsample-ratio-video 3 \
    --max-dec-tokens 32 \
    --num-beams 15
```

## Key Differences from AV-HuBERT

1. **Feature Dimension**: Auto-AVSR outputs 512-dimensional features vs AV-HuBERT's 1024-dimensional features
2. **No LoRA Support**: Auto-AVSR encoder is used as a frozen feature extractor (no `--use-lora-avhubert` or `lora_avhubert` in `--unfrozen_modules`)
3. **Pretrained Path**: Use `--pretrain-auto-avsr-path` instead of `--pretrain-avhubert-enc-video-path`

## Recommended Auto-AVSR Models

For best results, use these pretrained models from the Auto-AVSR model zoo:

- **VSR**: `vsr_trlrs3vox2_base.pth` (24.6% WER on LRS3)
- **AVSR**: `vsr_trlrs2lrs3vox2avsp_base.pth` (20.3% WER on LRS3)

## Notes

- Auto-AVSR models are used as frozen feature extractors
- The integration automatically handles the different feature dimensions
- All other training parameters remain the same as with AV-HuBERT