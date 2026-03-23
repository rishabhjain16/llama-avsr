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
        
        # Load model and processor
        model = AutoModel.from_pretrained(model_name)
        processor = AutoVideoProcessor.from_pretrained(model_name)
        
        # Get embedding dimension from model config
        embed_dim = model.config.hidden_size
        
        print(f"✓ Successfully loaded V-JEPA 2")
        print(f"  - Model: {model_name}")
        print(f"  - Embedding dim: {embed_dim}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Wrap in our adapter
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
        Preprocess videos for V-JEPA 2 using the official processor.
        
        Args:
            videos: torch.Tensor of shape (B, T, C, H, W)
                    where C=1 (grayscale) for LRS3 data
        
        Returns:
            Processed inputs ready for V-JEPA 2
        """
        B, T, C, H, W = videos.shape
        
        # Convert to list of numpy arrays for processor
        videos_list = []
        for i in range(B):
            # Convert single video to numpy (T, C, H, W)
            video = videos[i].cpu().to(torch.float32).numpy()  # (T, C, H, W)
            video = video.transpose(0, 2, 3, 1)  # (T, H, W, C)
            
            # Convert grayscale to RGB if needed
            if C == 1:
                video = video.repeat(3, axis=-1)  # (T, H, W, 3)
            
            # Normalize to [0, 255] range if needed
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            
            # Convert to list of frames
            frames = [video[t] for t in range(T)]
            videos_list.append(frames)
        
        # Process all videos
        processed = self.processor(videos_list, return_tensors="pt")
        
        return processed
    
    def forward(self, videos):
        """
        Extract V-JEPA 2 features from videos.
        
        Args:
            videos: torch.Tensor of shape (B, T, C, H, W)
        
        Returns:
            torch.Tensor of shape (B, T', embed_dim)
        """
        device = videos.device
        
        # Preprocess videos using official processor
        processed = self.preprocess_video(videos)
        
        # Debug: check what keys are available
        # print(f"Processor output keys: {processed.keys() if hasattr(processed, 'keys') else type(processed)}")
        
        # Get pixel values - try different possible keys
        if 'pixel_values_videos' in processed:
            pixel_values = processed['pixel_values_videos']
        elif hasattr(processed, 'pixel_values'):
            pixel_values = processed.pixel_values
        elif 'pixel_values' in processed:
            pixel_values = processed['pixel_values']
        elif isinstance(processed, dict) and len(processed) > 0:
            # Use first available tensor
            pixel_values = list(processed.values())[0]
        else:
            # Assume processed is already the tensor
            pixel_values = processed
        
        pixel_values = pixel_values.to(device)
        
        # Extract features
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model(pixel_values, output_hidden_states=True)
            
            # Get the last hidden state
            # V-JEPA 2 outputs: (B, num_patches, embed_dim)
            features = outputs.last_hidden_state
        
        # V-JEPA 2 outputs spatial-temporal tokens
        # We need temporal features for speech recognition
        # Average over spatial dimensions to get temporal sequence
        
        B, N, D = features.shape
        
        # For video transformers with patch_size=16, tubelet_size=2:
        # If input is 256x256 with 16 frames:
        # - Spatial patches: (256/16)^2 = 256
        # - Temporal patches: 16/2 = 8
        # - Total: 256 * 8 = 2048 tokens
        
        # Reshape to (B, T, H*W, D) and average spatial
        try:
            # Try to infer temporal dimension
            # Assuming square spatial layout
            img_size = self.processor.size.get('height', 256) if hasattr(self.processor, 'size') else 256
            patch_size = 16
            tubelet_size = 2
            
            spatial_patches = (img_size // patch_size) ** 2
            temporal_patches = N // spatial_patches
            
            if N == spatial_patches * temporal_patches:
                features = features.view(B, temporal_patches, spatial_patches, D)
                features = features.mean(dim=2)  # (B, T, D)
            else:
                # Fallback: use CLS token or mean pooling
                print(f"Warning: Could not reshape {N} tokens, using mean pooling")
                features = features.mean(dim=1, keepdim=True)  # (B, 1, D)
        except Exception as e:
            print(f"Warning: Feature reshaping failed ({e}), using mean pooling")
            features = features.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        return features


class VJEPA2Preprocessor(nn.Module):
    """
    Standalone preprocessor for V-JEPA 2 that can be loaded via hub.
    """
    def __init__(self):
        super().__init__()
        self.target_size = 256
    
    def __call__(self, videos):
        """
        Args:
            videos: torch.Tensor (B, T, C, H, W) or list of videos
        Returns:
            torch.Tensor (B, C, T, H, W) ready for V-JEPA 2
        """
        if isinstance(videos, list):
            videos = torch.stack(videos)
        
        B, T, C, H, W = videos.shape
        
        # Convert grayscale to RGB
        if C == 1:
            videos = videos.repeat(1, 1, 3, 1, 1)
        
        # Resize
        videos_flat = videos.view(B * T, 3, H, W)
        videos_resized = F.interpolate(
            videos_flat,
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        )
        videos_resized = videos_resized.view(B, T, 3, self.target_size, self.target_size)
        
        # Normalize
        if videos_resized.max() > 1.0:
            videos_resized = videos_resized / 255.0
        
        # Permute to (B, C, T, H, W)
        videos_resized = videos_resized.permute(0, 2, 1, 3, 4)
        
        return videos_resized


# Test function
if __name__ == "__main__":
    print("Testing V-JEPA 2 encoder integration...")
    
    # Create dummy video input (B=2, T=16, C=1, H=88, W=88) - typical LRS3 mouth crop
    dummy_video = torch.randn(2, 16, 1, 88, 88)
    
    print(f"Input shape: {dummy_video.shape}")
    
    # Test loading encoder
    try:
        encoder = load_vjepa2_video_encoder('facebook/vjepa2-vitl-fpc64-256')
        encoder.eval()
        
        with torch.no_grad():
            features = encoder(dummy_video)
        
        print(f"Output shape: {features.shape}")
        print(f"Feature dimension: {encoder.embed_dim}")
        print("✓ V-JEPA 2 encoder integration test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
