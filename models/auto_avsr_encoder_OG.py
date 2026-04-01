#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-AVSR Video Encoder Integration for Llama-AVSR
Adapted from auto-AVSR repository
"""

import torch
import torch.nn as nn
    
import sys
sys.path.append('auto_avsr')
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
    
def conv3x3(in_planes, out_planes, stride=1):
    """conv3x3.
    :param in_planes: int, number of channels in the input sequence.
    :param out_planes: int,  number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Sequential(
        nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = nn.SiLU(inplace=True)
            self.relu2 = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        relu_type="swish",
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.
        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class AutoAVSRVideoEncoder(nn.Module):
    """Auto-AVSR Video Encoder for feature extraction"""

    def __init__(self, relu_type="swish"):
        """__init__.
        :param relu_type: str, activation function used in the video front-end.
        """
        super(AutoAVSRVideoEncoder, self).__init__()

        self.frontend_nout = 64
        self.trunk = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            relu_type=relu_type,
        )

        # -- frontend3D
        if relu_type == "relu":
            frontend_relu = nn.ReLU(True)
        elif relu_type == "prelu":
            frontend_relu = nn.PReLU(self.frontend_nout)
        elif relu_type == "swish":
            frontend_relu = nn.SiLU(inplace=True)

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

    def forward(self, xs_pad):
        """forward.
        :param xs_pad: torch.Tensor, batch of padded input sequences (B, T, C, H, W).
        """
        # Transpose from (B, T, C, H, W) to (B, C, T, H, W) as in original auto-AVSR
        xs_pad = xs_pad.transpose(2, 1)
        
        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        Tnew = xs_pad.shape[2]  # output should be B x C2 x Tnew x H x W
        xs_pad = threeD_to_2D_tensor(xs_pad)
        xs_pad = self.trunk(xs_pad)
        xs_pad = xs_pad.view(B, Tnew, xs_pad.size(1))
        return xs_pad


class AutoAVSRFullEncoder(nn.Module):
    """Full Auto-AVSR encoder including frontend + projection + conformer encoder"""
    
    def __init__(self):
        super().__init__()
        # Frontend (3D CNN + ResNet)
        self.frontend = AutoAVSRVideoEncoder()
        
        # Projection layer (512 -> 768)
        self.proj_encoder = nn.Linear(512, 768)
        
        # We'll load the conformer encoder from the pretrained model
        self.encoder = None
        
    def forward(self, xs_pad):
        # Extract visual features using frontend
        xs_pad = self.frontend(xs_pad)  # (B, T, 512)
        
        # Project to encoder dimension
        xs_pad = self.proj_encoder(xs_pad)  # (B, T, 768)
        
        # Pass through conformer encoder if available
        if self.encoder is not None:
            # Create padding mask (assuming no padding for now)
            padding_mask = torch.ones(xs_pad.size(0), xs_pad.size(1)).to(xs_pad.device).unsqueeze(-2)
            xs_pad, _ = self.encoder(xs_pad, padding_mask)  # (B, T, 768)
        
        return xs_pad

def load_auto_avsr_video_encoder(pretrained_model_path):
    print(f"Loading Auto-AVSR encoder components from {pretrained_model_path}")

    ckpt = torch.load(pretrained_model_path, map_location='cpu')
    if "model" in ckpt:
        ckpt = ckpt["model"]
    
    encoder = AutoAVSRFullEncoder()
    
    frontend_state_dict = {}
    trunk_state_dict = {}
    proj_state_dict = {}
    conformer_state_dict = {}
    
    for key, value in ckpt.items():
        # --- 1. Frontend 3D ---
        if "frontend.frontend3D." in key:
            new_key = key.split("frontend3D.")[-1]
            frontend_state_dict[new_key] = value
            
        # --- 2. ResNet Trunk ---
        elif "frontend.trunk." in key:
            new_key = key.split("trunk.")[-1]
            trunk_state_dict[new_key] = value
            
        # --- 3. The Correct Projection Layer (512 -> 768) ---
        # We strictly look for encoder.embed.0 OR proj_encoder
        # and verify the shape matches [768, 512]
        elif "encoder.embed.0." in key or "proj_encoder." in key:
            if value.dim() > 1 and value.shape[0] == 768 and value.shape[1] == 512:
                new_key = key.split(".")[-1] # weight
                proj_state_dict[new_key] = value
                print(f"🎯 Matched Projector: {key} with shape {value.shape}")
            elif value.dim() == 1 and value.shape[0] == 768:
                new_key = key.split(".")[-1] # bias
                proj_state_dict[new_key] = value

        # --- 4. Conformer Blocks ---
        elif key.startswith("encoder.") and not any(x in key for x in ["frontend", "embed"]):
            new_key = key.replace('encoder.', '')
            conformer_state_dict[new_key] = value

    # Load with strict=False to handle the specific sub-modules
    encoder.frontend.frontend3D.load_state_dict(frontend_state_dict)
    encoder.frontend.trunk.load_state_dict(trunk_state_dict)
    encoder.proj_encoder.load_state_dict(proj_state_dict)
    
    # Initialize and load Conformer
    encoder.encoder = ConformerEncoder(
        attention_dim=768,
        attention_heads=12,
        linear_units=3072,
        num_blocks=12,
        cnn_module_kernel=31,
    )
    encoder.encoder.load_state_dict(conformer_state_dict)
    
    return encoder
# def load_auto_avsr_video_encoder(pretrained_model_path):
#     """Load Auto-AVSR encoder components (frontend + projection + encoder) for feature extraction"""
    
#     print(f"Loading Auto-AVSR encoder components from {pretrained_model_path}")

#     # Load checkpoint
#     ckpt = torch.load(pretrained_model_path, map_location='cpu')
    
#     # Create our encoder
#     encoder = AutoAVSRFullEncoder()
    
#     # Load frontend components directly from checkpoint
#     frontend_state_dict = {}
#     trunk_state_dict = {}
#     proj_state_dict = {}
#     encoder_state_dict = {}
    
#     for key, value in ckpt.items():
#         if key.startswith('frontend.frontend3D.'):
#             new_key = key.replace('frontend.frontend3D.', '')
#             frontend_state_dict[new_key] = value
#         elif key.startswith('frontend.trunk.'):
#             new_key = key.replace('frontend.trunk.', '')
#             trunk_state_dict[new_key] = value
#         elif key.startswith('proj_encoder.'):
#             new_key = key.replace('proj_encoder.', '')
#             proj_state_dict[new_key] = value
#         elif key.startswith('encoder.'):
#             new_key = key.replace('encoder.', '')
#             encoder_state_dict[new_key] = value
    
#     # Load frontend weights
#     encoder.frontend.frontend3D.load_state_dict(frontend_state_dict)
#     encoder.frontend.trunk.load_state_dict(trunk_state_dict)
#     print("Loaded frontend components")
    
#     # Load projection layer
#     encoder.proj_encoder.load_state_dict(proj_state_dict)
#     print("Loaded projection layer")
    
#     # Create and load conformer encoder
#     encoder.encoder = ConformerEncoder(
#         attention_dim=768,
#         attention_heads=12,
#         linear_units=3072,
#         num_blocks=12,
#         cnn_module_kernel=31,
#     )
#     encoder.encoder.load_state_dict(encoder_state_dict)
#     print("Loaded conformer encoder")
    
#     print("Successfully loaded Auto-AVSR encoder components (frontend + projection + encoder)")
#     return encoder

# Example usage
if __name__ == "__main__":
    import torch
    import torchvision

    checkpoint_path = "/home/rishabh/Desktop/Experiments/Llama-AVSR/ckps/vsr_trlrs2lrs3vox2avsp_base.pth"
    extractor = load_auto_avsr_video_encoder(checkpoint_path)
    extractor.eval()  # set to eval mode

    def load_video_tensor(video_path):
        """
        Load a video file as a torch tensor suitable for Auto-AVSR encoder.
        Returns: B x T x C x H x W (C=1 grayscale)
        """
        # Read video (T, H, W, C)
        video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
        
        # Convert to float and normalize
        video_frames = video_frames.float() / 255.0  # T x H x W x C
        
        # Convert RGB to grayscale manually
        video_gray = (
            0.2989 * video_frames[..., 0] +
            0.5870 * video_frames[..., 1] +
            0.1140 * video_frames[..., 2]
        )  # T x H x W

        # Add channel dimension
        video_gray = video_gray.unsqueeze(-1)  # T x H x W x 1

        # Permute to T x C x H x W
        video_tensor = video_gray.permute(0, 3, 1, 2)

        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)  # B x T x C x H x W

        return video_tensor

    # --- Test with a real video ---
    video_path = "/home/rishabh/Desktop/Datasets/lrs3_rf/lrs3/lrs3_video_seg16s/test/0gks6ceq4eQ/00005.mp4"
    video_tensor = load_video_tensor(video_path)

    with torch.no_grad():
        features = extractor(video_tensor)

    print("Encoded features shape:", features.shape)  # B x T
