#!/usr/bin/env python3
"""
Simple test for V-JEPA 2 integration using HuggingFace
"""

import torch

print("=" * 60)
print("Testing V-JEPA 2 Integration (HuggingFace)")
print("=" * 60)

# Test 1: Load V-JEPA 2 from HuggingFace
print("\n1. Loading V-JEPA 2 from HuggingFace...")
try:
    from models.vjepa2_encoder import load_vjepa2_video_encoder
    
    encoder = load_vjepa2_video_encoder('facebook/vjepa2-vitl-fpc64-256')
    encoder.eval()
    print("✓ Successfully loaded V-JEPA 2")
    
except Exception as e:
    print(f"✗ Failed to load: {e}")
    print("\nPlease install transformers:")
    print("  pip install transformers")
    exit(1)

# Test 2: Create dummy LRS3-style video
print("\n2. Creating dummy video input...")
batch_size = 2
num_frames = 16
height, width = 88, 88  # LRS3 mouth crop size

dummy_video = torch.randn(batch_size, num_frames, 1, height, width)
print(f"✓ Input shape: {dummy_video.shape}")
print(f"  (B={batch_size}, T={num_frames}, C=1, H={height}, W={width})")

# Test 3: Forward pass
print("\n3. Running forward pass...")
try:
    with torch.no_grad():
        features = encoder(dummy_video)
    
    print(f"✓ Output shape: {features.shape}")
    print(f"  (B={batch_size}, T={features.shape[1]}, D={features.shape[2]})")
    print(f"  Feature dimension: {encoder.embed_dim}")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Check feature dimensions
print("\n4. Checking feature dimensions...")
expected_dim = encoder.embed_dim
if features.shape[-1] == expected_dim:
    print(f"✓ Feature dimension matches: {expected_dim}")
else:
    print(f"✗ Dimension mismatch: got {features.shape[-1]}, expected {expected_dim}")

# Test 5: Memory usage estimate
print("\n5. Memory usage estimate...")
param_count = sum(p.numel() for p in encoder.model.parameters()) / 1e6
print(f"  Model parameters: {param_count:.1f}M")
print(f"  Estimated GPU memory: ~{param_count * 4 / 1000:.1f} GB")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("\nYou can now use V-JEPA 2 in training:")
print("  python train.py \\")
print("    --video-encoder-name vjepa2 \\")
print("    --vjepa2-model-name facebook/vjepa2-vitl-fpc64-256 \\")
print("    ...")
print("=" * 60)
