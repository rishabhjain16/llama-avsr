#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-JEPA 2 Video Encoder Integration for Llama-AVSR
Uses HuggingFace transformers for proper checkpoint loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_vjepa2_video_encoder(model_name='facebook/vjepa2-vitl-fpc64-256'):
    """
    Load V-JEPA 2 encoder for video feature extraction via HuggingFace.
    
    Args:
        model_name: str, HuggingFace model identifier
            Options:
            - 'facebook/vjepa2-vitl-fpc64-256' (ViT-L, 1024 dim)
            - 'facebook/vjepa2-vith-fpc64-256' (ViT-H, 1280 dim)
            - 'facebook/vjepa2-vitg-fpc64-256' (ViT-g, 1408 dim)
            - 'facebook/vjepa2-vitg-fpc64-384' (ViT-g, 1408 dim, 384px)
    
    Returns:
        VJEPA2VideoEncoder instance
    """
    print(f"Loading V-JEPA 2 from HuggingFace: {model_name}")
    
    try:
        from transformers import AutoModel, AutoVideoProcessor
        
        model = AutoModel.from_pretrained(model_name)
        processor = AutoVideoProcessor.from_pretrained(model_name)
        embed_dim = model.config.hidden_size
        
        print(f"✓ Successfully loaded V-JEPA 2")
        print(f"  - Model: {model_name}")
        print(f"  - Embedding dim: {embed_dim}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        return VJEPA2VideoEncoder(model, processor, embed_dim)
        
    except Exception as e:
        print(f"✗ Failed to load from HuggingFace: {e}")
        print("\nPlease install transformers:")
        print("  pip install transformers")
        print("\nOr check model name. Available models:")
        print("  - facebook/vjepa2-vitl-fpc64-256")
        print("  - facebook/vjepa2-vith-fpc64-256")
        print("  - facebook/vjepa2-vitg-fpc64-256")
        print("  - facebook/vjepa2-vitg-fpc64-384")
        raise


class VJEPA2VideoEncoder(nn.Module):
    """
    Adapter for V-JEPA 2 encoder to work with Llama-AVSR pipeline.
    Handles input preprocessing and feature extraction using HuggingFace models.
    """
    
    def __init__(self, vjepa2_model, vjepa2_processor, embed_dim):
        super().__init__()
        self.model = vjepa2_model
        self.processor = vjepa2_processor
        self.embed_dim = embed_dim

    def preprocess_video(self, videos):
        """
        Preprocess videos for V-JEPA 2 using the official HF processor.
        Pads or uniformly samples to exactly 64 frames (fpc64 requirement).
        
        Args:
            videos: torch.Tensor of shape (B, T, C, H, W), C=1 grayscale OK
        
        Returns:
            Processed inputs ready for V-JEPA 2
        """
        B, T, C, H, W = videos.shape

        # FIX 1: normalise to exactly 64 frames
        TARGET_FRAMES = 64
        if T < TARGET_FRAMES:
            pad = videos[:, -1:].repeat(1, TARGET_FRAMES - T, 1, 1, 1)
            videos = torch.cat([videos, pad], dim=1)
        elif T > TARGET_FRAMES:
            indices = torch.linspace(0, T - 1, TARGET_FRAMES).long()
            videos = videos[:, indices]
        
        B, T, C, H, W = videos.shape  # T == 64 now

        # Convert to list of numpy arrays for the HF processor
        videos_list = []
        for i in range(B):
            video = videos[i].cpu().to(torch.float32).numpy()  # (T, C, H, W)
            video = video.transpose(0, 2, 3, 1)                # (T, H, W, C)
            
            if C == 1:
                video = video.repeat(3, axis=-1)               # (T, H, W, 3)
            
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            
            frames = [video[t] for t in range(T)]
            videos_list.append(frames)
        
        processed = self.processor(videos_list, return_tensors="pt")
        return processed

    def forward(self, videos):
        """
        Extract V-JEPA 2 features from videos.
        
        Args:
            videos: torch.Tensor of shape (B, T, C, H, W)
        
        Returns:
            torch.Tensor of shape (B, T_temporal, embed_dim)
        """
        device = videos.device
        
        processed = self.preprocess_video(videos)
        
        # Extract pixel values from processor output
        if 'pixel_values_videos' in processed:
            pixel_values = processed['pixel_values_videos']
        elif hasattr(processed, 'pixel_values'):
            pixel_values = processed.pixel_values
        elif 'pixel_values' in processed:
            pixel_values = processed['pixel_values']
        elif isinstance(processed, dict) and len(processed) > 0:
            pixel_values = list(processed.values())[0]
        else:
            pixel_values = processed
        
        pixel_values = pixel_values.to(device)

        # FIX 2: updated autocast API
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # FIX 3: use keyword arg — HF V-JEPA 2 requires pixel_values_videos=
            outputs = self.model(pixel_values_videos=pixel_values, output_hidden_states=True)
            features = outputs.last_hidden_state  # (B, N, D)
        
        B, N, D = features.shape
        
        img_size = self.processor.size.get('height', 256) if hasattr(self.processor, 'size') else 256
        patch_size = 16
        tubelet_size = 2
        
        spatial_patches = (img_size // patch_size) ** 2
        temporal_patches = N // spatial_patches
        
        if N == spatial_patches * temporal_patches and temporal_patches > 1:
            features = features.view(B, temporal_patches, spatial_patches, D)
            features = features.mean(dim=2)  # (B, T_temporal, D)
        else:
            print(f"Warning: Could not reshape {N} tokens, using mean pooling")
            features = features.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        return features


# Test function
if __name__ == "__main__":
    print("Testing V-JEPA 2 encoder integration...")
    
    for T in [16, 21, 45, 64, 66]:
        dummy_video = torch.randn(2, T, 1, 88, 88)
        print(f"\nInput shape: {dummy_video.shape}")
        
        try:
            encoder = load_vjepa2_video_encoder('facebook/vjepa2-vitl-fpc64-256')
            encoder.eval()
            
            with torch.no_grad():
                features = encoder(dummy_video)
            
            print(f"Output shape: {features.shape}")  # expect (2, 32, 1024)
            assert features.shape == (2, 32, 1024), f"Unexpected shape: {features.shape}"
            print("✓ Passed")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()