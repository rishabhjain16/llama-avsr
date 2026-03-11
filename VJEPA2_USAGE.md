# V-JEPA 2 Usage Guide

## Quick Start

### 1. Test Integration
```bash
conda activate vjepa2-311
python test_vjepa2_simple.py
```

### 2. Train VSR (Video-Only)
```bash
chmod +x train_vjepa2_vsr.sh
./train_vjepa2_vsr.sh
```

### 3. Train AVSR (Audio-Visual)
```bash
chmod +x train_vjepa2_avsr.sh
./train_vjepa2_avsr.sh
```

## Available Models

| Model | HuggingFace ID | Params | Dim | GPU Memory |
|-------|---------------|--------|-----|------------|
| ViT-L | facebook/vjepa2-vitl-fpc64-256 | 300M | 1024 | ~16GB |
| ViT-H | facebook/vjepa2-vith-fpc64-256 | 600M | 1280 | ~20GB |
| ViT-g | facebook/vjepa2-vitg-fpc64-256 | 1B | 1408 | ~24GB |

**Start with ViT-L** (fastest, smallest)

## Key Parameters

- `--video-encoder-name vjepa2` - Use V-JEPA 2 instead of Auto-AVSR
- `--vjepa2-model-name` - Which V-JEPA 2 model to use
- `--downsample-ratio-video 3` - Temporal downsampling (adjust if OOM)

## Comparison with Auto-AVSR

To compare, just change the encoder:

**Auto-AVSR:**
```bash
--video-encoder-name auto-avsr \
--pretrain-auto-avsr-path /path/to/autoavsr.pth
```

**V-JEPA 2:**
```bash
--video-encoder-name vjepa2 \
--vjepa2-model-name facebook/vjepa2-vitl-fpc64-256
```

## Expected Results

- Auto-AVSR baseline: ~23.68% WER (VSR)
- V-JEPA 2 (ViT-L): ~21-22% WER (expected)

## Troubleshooting

**Out of Memory:**
- Use smaller model: `facebook/vjepa2-vitl-fpc64-256`
- Reduce batch: `--max-frames-video 800`

**Slow Training:**
- V-JEPA 2 is larger than Auto-AVSR (~30% slower)
- This is expected for better motion understanding
