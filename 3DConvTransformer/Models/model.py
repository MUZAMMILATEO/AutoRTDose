from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Note: The original pvt_v2.py is not used, but we keep the structure for compatibility
# with potential existing imports or if you wish to re-introduce the PVTv2 later.
# If you are placing this file in a directory named 'Models', you may need: 
# from . import pvt_v2 
# For now, we assume you've replaced pvt_v2.py content.
# from Models import pvt_v2 # Remove this line as it is not used


# ------------------------------
# 3D Residual Block (RB)
# ------------------------------
class RB(nn.Module):
    """Residual Block with Conv3d for 3D processing."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Input layers use Conv3d
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        # Output layers use Conv3d
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # Skip connection uses Conv3d if channel count changes
        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


# ------------------------------
# 3D Fully Convolutional Branch (FCB) - U-Net style encoder/decoder
# ------------------------------
class FCB(nn.Module):
    """The 3D CNN component for dense feature extraction."""
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=128,
    ):

        super().__init__()

        # Initial 3D Conv - uses the dynamically passed in_channels
        self.enc_blocks = nn.ModuleList(
            [nn.Conv3d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels)) 
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                # 3D Downsampling (Conv3d with stride=2)
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv3d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(), 
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    # 3D Upsampling
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv3d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        # Input: (N, C, D, H, W)
        hs = []
        h = x
        # Encoder
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        
        # Middle
        h = self.middle_block(h)
        
        # Decoder
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1) 
            h = module(cat_in)
            
        # Output: (N, 32, D, H, W)
        return h


# ---------------------------------------------
# Transformer Branch (TB) - Z-Axis Modulator
# ---------------------------------------------
class TB(nn.Module):
    """Z-Axis Transformer for inter-slice attention and channel modulation."""
    def __init__(self, in_channels=3, seq_len=128, fc_channels=32, num_layers=4, num_heads=4):
        super().__init__()
        
        compression_channels = in_channels # Use the passed in_channels
        target_spatial_size = 8 # Drastically reduce spatial size to 8x8

        # --- A. Pre-Transformer Embedding (Spatial Compression) ---
        # Reduce 128x128 to 8x8 while keeping 'in_channels'
        self.spatial_compress = nn.Sequential(
            # Output compression_channels, reducing 128x128 to 32x32 (Stride 4)
            nn.Conv3d(in_channels, compression_channels, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2)),
            nn.InstanceNorm3d(compression_channels), 
            nn.SiLU(),
            # Further reduce 32x32 to 8x8 (Stride 4)
            nn.Conv3d(compression_channels, compression_channels, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 2, 2)), 
            nn.InstanceNorm3d(compression_channels),
            nn.SiLU(),
        )
        
        # Output shape after compression: (N, C, D, 8, 8). Sequence feature size: C * 8 * 8
        transformer_input_dim = compression_channels * target_spatial_size * target_spatial_size
        transformer_output_dim = 64 # Feature size for the Transformer blocks
        
        # --- B. Z-Axis Transformer Setup ---
        
        # 1. Positional Encoding
        # The size of this parameter must match the seq_len (D dimension) passed in
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, transformer_input_dim))
        nn.init.trunc_normal_(self.pos_encoder, std=.02)

        # 2. Sequence Projection (Downscale the large feature vector)
        self.input_projection = nn.Linear(transformer_input_dim, transformer_output_dim) 

        # 3. Transformer Encoder (num_heads=4 works well with d_model=64)
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_output_dim, 
            nhead=num_heads, 
            dim_feedforward=transformer_output_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        # 4. Modulation Vector Projection (Project to match FCB channels: 64 -> 32)
        self.output_projection = nn.Linear(transformer_output_dim, fc_channels) 

    def forward(self, x):
        # Input x: (N, C, D, H, W)
        N, _, D, _, _ = x.shape
        
        # 1. Spatial Compression: (N, C, D, 128, 128) -> (N, C, D, 8, 8)
        x_embed = self.spatial_compress(x) 
        
        # 2. Flatten H'xW' into sequence vectors: (N, C, D, 8, 8) -> (N, D, C*H'*W')
        # Permute to (N, D, C, H', W') then flatten C, H', W'
        x_seq = x_embed.permute(0, 2, 1, 3, 4).reshape(N, D, -1) 
        
        # 3. Input Projection and Positional Encoding
        x_seq = x_seq + self.pos_encoder[:, :D, :] # Use only up to D slices
        x_seq = self.input_projection(x_seq) 
        
        # 4. Z-Axis Attention
        z_out = self.transformer_encoder(x_seq) 
        
        # 5. Modulation Vector Projection
        z_modulation = self.output_projection(z_out) 
        
        return z_modulation


# ------------------------------
# 3D FCBFormer (The main model)
# ------------------------------
class FCBFormer(nn.Module):
    """The combined 3D CNN (FCB) and Z-Axis Transformer (TB) model."""
    # FIX: Updated signature to accept in_channels, out_channels, and size
    def __init__(self, in_channels, out_channels, size): 
        super().__init__()
        
        # FCB_OUT_CHANNELS determines the channel count for the U-Net bottleneck/fusion (32)
        FCB_OUT_CHANNELS = 32
        
        # Pass in_channels (3) to FCB
        self.FCB = FCB(in_channels=in_channels, min_level_channels=FCB_OUT_CHANNELS) 
        
        # Pass in_channels (3) and size (128) to TB
        self.TB = TB(in_channels=in_channels, seq_len=size, fc_channels=FCB_OUT_CHANNELS) 
        
        # Final Prediction Head (3D) - processes the modulated FCB output (32 channels)
        self.PH = nn.Sequential(
            RB(FCB_OUT_CHANNELS, FCB_OUT_CHANNELS),
            RB(FCB_OUT_CHANNELS, FCB_OUT_CHANNELS),
            # Use out_channels (1) for the final convolution to output the dose map
            nn.Conv3d(FCB_OUT_CHANNELS, out_channels, kernel_size=1) 
        )

    def forward(self, x):
        # Input x: (N, 3, 128, 128, 128)
        
        # --- 1. FCB Path (3D CNN) ---
        x_FCB = self.FCB(x) # Output: (N, 32, D, H, W)

        # --- 2. TB Path (Z-Axis Modulator) ---
        x_TB = self.TB(x) # Output: (N, D, 32)

        # --- 3. Fusion (Slice-wise Channel Modulation) ---
        # Reshape Z-Modulator: (N, D, 32) -> (N, 32, D, 1, 1)
        z_mod_reshaped = x_TB.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        
        # Multiplication (Broadcasting)
        x_fused = x_FCB * z_mod_reshaped 

        # --- 4. Final Prediction ---
        out = self.PH(x_fused) # Output: (N, 1, D, H, W)
        
        return out
