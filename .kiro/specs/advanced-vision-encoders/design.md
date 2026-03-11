# Design Document

## Overview

This design extends the existing AVSR-LLM system to support advanced vision encoders (CLIP and ViViT) while maintaining the current modular architecture. The system currently supports AV-HuBERT and Auto-AVSR encoders through a unified interface in the `encode_video` method. We will extend this pattern to accommodate CLIP and ViViT encoders with their specific requirements for input preprocessing, feature extraction, and output formatting.

The key design principle is to maintain backward compatibility while providing a clean, extensible interface for new vision encoders. Each encoder will be responsible for handling its own input format requirements and producing a standardized output format of (B, T, D) where B is batch size, T is temporal dimension, and D is feature dimension.

## Architecture

### Current Architecture Analysis

The existing system follows this pattern:
1. **Encoder Initialization**: Each encoder is initialized in the `AVSR_LLMs.__init__()` method based on `video_encoder_name`
2. **Feature Extraction**: The `encode_video()` method handles encoder-specific processing
3. **Dimension Projection**: Features are projected to LLM hidden dimension via `video_proj`
4. **Temporal Downsampling**: Optional downsampling based on `downsample_ratio_video`

### Extended Architecture

We will extend this architecture to support CLIP and ViViT:

```
Input Video (B, T, C, H, W)
    ↓
Encoder-Specific Preprocessing
    ↓
Vision Encoder (CLIP/ViViT/AV-HuBERT/Auto-AVSR)
    ↓
Feature Extraction (B, T, D_encoder)
    ↓
Temporal Downsampling (optional)
    ↓
Projection to LLM Hidden Dim (B, T_down, D_llm)
```

## Components and Interfaces

### 1. Vision Encoder Factory

We will create a factory pattern to handle encoder initialization:

```python
class VisionEncoderFactory:
    @staticmethod
    def create_encoder(encoder_name, config):
        if encoder_name == "clip":
            return CLIPVideoEncoder(config)
        elif encoder_name == "vivit":
            return ViViTVideoEncoder(config)
        # ... existing encoders
```

### 2. Base Vision Encoder Interface

All encoders will implement a common interface:

```python
class BaseVideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, videos):
        """
        Args:
            videos: Tensor of shape (B, T, C, H, W)
        Returns:
            features: Tensor of shape (B, T_out, D_out)
        """
        raise NotImplementedError
        
    @property
    def output_dim(self):
        """Returns the output feature dimension"""
        raise NotImplementedError
        
    def preprocess_input(self, videos):
        """Encoder-specific input preprocessing"""
        return videos
```

### 3. CLIP Video Encoder

CLIP requires special handling since it's designed for single images:

```python
class CLIPVideoEncoder(BaseVideoEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.clip_model = clip.load(config.model_name)[0]
        self.clip_model.requires_grad_(False)
        
    def forward(self, videos):
        # Process each frame independently
        B, T, C, H, W = videos.shape
        videos_flat = videos.view(B * T, C, H, W)
        
        # CLIP preprocessing
        videos_preprocessed = self.preprocess_input(videos_flat)
        
        # Extract features
        with torch.no_grad():
            features = self.clip_model.encode_image(videos_preprocessed)
            
        # Reshape back to temporal format
        features = features.view(B, T, -1)
        return features
        
    @property
    def output_dim(self):
        return self.clip_model.visual.output_dim
```

### 4. ViViT Video Encoder

ViViT natively handles video sequences:

```python
class ViViTVideoEncoder(BaseVideoEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.vivit_model = self._load_vivit_model(config)
        self.vivit_model.requires_grad_(False)
        
    def forward(self, videos):
        # ViViT expects (B, T, C, H, W)
        videos_preprocessed = self.preprocess_input(videos)
        
        with torch.no_grad():
            features = self.vivit_model.forward_features(videos_preprocessed)
            
        return features
```

### 5. Enhanced AVSR_LLMs Integration

The main model class will be updated to support the new encoders:

```python
class AVSR_LLMs(nn.Module):
    def __init__(self, ..., video_encoder_config=None):
        # ... existing initialization
        
        if modality in ["video", "audiovisual"]:
            self.video_encoder = VisionEncoderFactory.create_encoder(
                self.video_encoder_name, 
                video_encoder_config or {}
            )
            video_dim = self.video_encoder.output_dim
            
            self.video_proj = nn.Sequential(
                nn.Linear(video_dim * self.downsample_ratio_video, intermediate_size),
                nn.ReLU(),
                nn.Linear(intermediate_size, hidden_size)
            )
    
    def encode_video(self, videos):
        video_enc = self.video_encoder(videos)
        
        # Apply temporal downsampling if needed
        if self.downsample_ratio_video != 1:
            video_enc = self._downsample_temporal(video_enc)
            
        return video_enc
```

## Data Models

### Configuration Classes

```python
@dataclass
class CLIPConfig:
    model_name: str = "ViT-B/32"  # CLIP model variant
    image_size: int = 224
    normalize: bool = True
    
@dataclass
class ViViTConfig:
    model_name: str = "vivit-b-16x2"
    num_frames: int = 16
    image_size: int = 224
    patch_size: int = 16
    
@dataclass
class VideoEncoderConfig:
    encoder_name: str
    clip_config: Optional[CLIPConfig] = None
    vivit_config: Optional[ViViTConfig] = None
```

### Input/Output Specifications

- **Input Format**: All encoders receive videos as `(B, T, C, H, W)` tensors
- **Output Format**: All encoders produce features as `(B, T_out, D_out)` tensors
- **Preprocessing**: Each encoder handles its own input preprocessing requirements
- **Temporal Alignment**: Output temporal dimension should align with audio features when possible

## Error Handling

### 1. Encoder Loading Errors
- Missing dependencies (transformers, clip-by-openai, etc.)
- Invalid model configurations
- Insufficient GPU memory

### 2. Runtime Errors
- Input format mismatches
- Dimension incompatibilities
- Memory allocation failures

### 3. Configuration Errors
- Invalid encoder names
- Incompatible parameter combinations
- Missing required configuration fields

### Error Recovery Strategies
- Graceful fallback to default configurations
- Clear error messages with troubleshooting steps
- Validation of inputs before processing
- Memory optimization suggestions

## Testing Strategy

### 1. Unit Tests
- Individual encoder functionality
- Input/output shape validation
- Configuration parsing
- Error handling scenarios

### 2. Integration Tests
- End-to-end feature extraction pipeline
- Compatibility with existing audio encoders
- Memory usage profiling
- Performance benchmarking

### 3. Regression Tests
- Backward compatibility with existing encoders
- Model loading and saving
- Training/inference consistency

### 4. Performance Tests
- Throughput comparison between encoders
- Memory usage analysis
- GPU utilization metrics
- Batch processing efficiency

## Implementation Considerations

### 1. Memory Management
- Efficient batch processing for CLIP (frame-by-frame)
- Gradient checkpointing for large models
- Optional mixed precision support

### 2. Temporal Handling
- CLIP: Aggregate frame-level features temporally
- ViViT: Native temporal modeling
- Consistent temporal downsampling across encoders

### 3. Model Loading
- Lazy loading to reduce memory footprint
- Caching of frequently used models
- Support for custom model paths

### 4. Preprocessing Pipeline
- Standardized input normalization
- Encoder-specific transformations
- Efficient tensor operations

### 5. Configuration Management
- YAML/JSON configuration files
- Environment variable overrides
- Runtime parameter validation