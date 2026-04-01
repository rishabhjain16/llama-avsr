import torch
import torch.nn as nn
import sys

# Ensure your path to auto_avsr is correct for these imports
try:
    from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
except ImportError:
    print("Warning: Could not import ConformerEncoder. Ensure 'auto_avsr' is in your PYTHONPATH.")

# --- 1D HELPER FUNCTIONS ---

def conv1d_3(in_planes, out_planes, stride=1):
    """1D convolution with kernel size 3."""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="swish"):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv1d_3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SiLU(inplace=True) if relu_type == "swish" else nn.ReLU(inplace=True)
        self.conv2 = conv1d_3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

# --- RESNET 1D (TEMPORAL TRUNK) ---

class ResNet1D(nn.Module):
    def __init__(self, block, layers, relu_type="swish"):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        # The 'Stem' - These were the "Unexpected Keys" in your error log
        self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.SiLU(inplace=True) if relu_type == "swish" else nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x # Returns (B, 512, T)

# --- VIDEO ENCODER ---

class AutoAVSRVideoEncoder(nn.Module):
    def __init__(self, relu_type="swish"):
        super(AutoAVSRVideoEncoder, self).__init__()
        self.frontend_nout = 64
        
        # 1D Temporal Trunk
        self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = nn.SiLU(inplace=True) if relu_type == "swish" else nn.ReLU(True)
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.transpose(1, 2)
        x = self.frontend3D(x) 
        
        # CRITICAL CHANGE: Collapse spatial H, W to make it a 1D Temporal signal
        # Output: (B, 64, T, H, W) -> (B, 64, T)
        x = x.mean(dim=[-2, -1]) 
        
        x = self.trunk(x) # (B, 512, T)
        return x.transpose(1, 2) # (B, T, 512)

# --- FULL ENCODER MODULE ---

class AutoAVSRFullEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = AutoAVSRVideoEncoder()
        self.proj_encoder = nn.Linear(512, 768)
        self.encoder = None
        
    def forward(self, xs_pad):
        xs_pad = self.frontend(xs_pad)
        xs_pad = self.proj_encoder(xs_pad)
        if self.encoder is not None:
            padding_mask = torch.ones(xs_pad.size(0), xs_pad.size(1)).to(xs_pad.device).unsqueeze(-2)
            xs_pad, _ = self.encoder(xs_pad, padding_mask)
        return xs_pad

# --- LOADING LOGIC ---
def load_auto_avsr_video_encoder(pretrained_model_path):
    # Add auto_avsr to path programmatically to fix the Import Warning
    import os
    sys.path.append(os.path.join(os.getcwd(), 'auto_avsr'))
    
    print(f"Loading Auto-AVSR components from {pretrained_model_path}")
    ckpt = torch.load(pretrained_model_path, map_location='cpu')
    if "model" in ckpt: ckpt = ckpt["model"]
    
    model = AutoAVSRFullEncoder()
    f3d_dict, trunk_dict, proj_dict, conf_dict = {}, {}, {}, {}
    
    for k, v in ckpt.items():
        # --- 1. Smart Hunt for Frontend 3D ---
        if "frontend3D." in k:
            # We only want the Video frontend, not the Audio one
            if "v_frontend" in k or "video" in k or "encoder.frontend" in k:
                new_key = k.split("frontend3D.")[-1]
                f3d_dict[new_key] = v
            
        # --- 2. Smart Hunt for ResNet Trunk ---
        elif "trunk." in k:
            # Check shape to ensure it's Video (kernel size 3) not Audio (80)
            if "v_frontend" in k or (v.dim() == 3 and v.shape[-1] == 3):
                new_key = k.split("trunk.")[-1]
                trunk_dict[new_key] = v
            
        # --- 3. Projection Layer (512 -> 768) ---
        elif any(p in k for p in ["proj_encoder.", "embed.0.", "proj."]):
            if v.dim() > 1 and v.shape == torch.Size([768, 512]):
                proj_dict["weight"] = v
            elif v.dim() == 1 and v.shape[0] == 768:
                proj_dict["bias"] = v

        # --- 4. Conformer Encoder ---
        elif "encoder." in k and not any(x in k for x in ["frontend", "embed", "a_frontend"]):
            conf_dict[k.replace('encoder.', '')] = v

    # Debug check: If these are empty, we know the search failed
    if not f3d_dict:
        print("❌ ERROR: Could not find Video Frontend weights in checkpoint!")
        print(f"Available keys (first 5): {list(ckpt.keys())[:5]}")
        return None

    # Load with strict=False to allow for slight variations in internal naming
    model.frontend.frontend3D.load_state_dict(f3d_dict, strict=False)
    model.frontend.trunk.load_state_dict(trunk_dict, strict=False)
    model.proj_encoder.load_state_dict(proj_dict, strict=False)
    
    # Initialize and load Conformer
    from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
    model.encoder = ConformerEncoder(
        attention_dim=768, attention_heads=12, linear_units=3072, num_blocks=12, cnn_module_kernel=31
    )
    model.encoder.load_state_dict(conf_dict, strict=False)
    
    print("✅ Successfully mapped Video weights from AV checkpoint.")
    return model