from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Note: The original pvt_v2.py is not used, but we keep the structure for compatibility
# with potential existing imports or if you wish to re-introduce the PVTv2 later.

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
    """The 3D CNN component for dense feature extraction (U-Net style)."""
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
                # The pop() gets the corresponding skip connection channel count
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
            # Concatenate current feature map with skip connection (from hs.pop())
            cat_in = torch.cat([h, hs.pop()], dim=1) 
            h = module(cat_in)
            
        # Output: (N, 32, D, H, W)
        return h


# ---------------------------------------------
# Transformer Branch (TB) - Directional Modulator
# ---------------------------------------------
# This class is now used for Z, Y, and X axes. It always assumes the 
# sequence dimension is at index 2 (D) and the spatial dims are at 3, 4 (H, W).
# Permutations are handled in FCBFormer.
class TB(nn.Module):
    """Directional Transformer for inter-slice attention and channel modulation."""
    def __init__(self, in_channels=3, seq_len=128, fc_channels=32, num_layers=4, num_heads=4):
        super().__init__()
        
        compression_channels = in_channels 
        target_spatial_size = 64 # Drastically reduce spatial size to 64x64
        stride_size = 2

        # --- A. Pre-Transformer Embedding (Spatial Compression) ---
        # Reduces the last two spatial dimensions (which will be H and W after permutation)
        self.spatial_compress = nn.Sequential(
            # Output compression_channels, reducing 128x128 to 32x32 (Stride 4)
            # The kernel (1, 5, 5) ensures no reduction along the sequence (D) axis
            nn.Conv3d(in_channels, compression_channels, kernel_size=(1, 5, 5), stride=(1, stride_size, stride_size), padding=(0, 2, 2)),
            nn.InstanceNorm3d(compression_channels), 
            nn.SiLU(),
            # Further reduce 32x32 to 8x8 (Stride 4)
            nn.Conv3d(compression_channels, compression_channels, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)), 
            nn.InstanceNorm3d(compression_channels),
            nn.SiLU(),
        )
        
        # Output shape after compression: (N, C, Seq_len, 8, 8). Sequence feature size: C * 8 * 8
        transformer_input_dim = compression_channels * target_spatial_size * target_spatial_size
        transformer_output_dim = 128 # 64 # Feature size for the Transformer blocks
        
        # --- B. Directional Transformer Setup ---
        
        # 1. Positional Encoding
        # The size of this parameter must match the seq_len (D, H, or W dimension, which is 'size')
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
        # Input x is assumed to be (N, C, Seq_len, Spatial1, Spatial2)
        N, _, D, _, _ = x.shape # D is now the sequence length (D, H, or W)
        
        # 1. Spatial Compression: (N, C, Seq, 128, 128) -> (N, C, Seq, 8, 8)
        x_embed = self.spatial_compress(x) 
        
        # 2. Flatten Spatial: (N, C, Seq, 8, 8) -> (N, Seq, C*8*8)
        # Permute to (N, Seq, C, H', W') then flatten C, H', W'
        x_seq = x_embed.permute(0, 2, 1, 3, 4).reshape(N, D, -1) 
        
        # 3. Input Projection and Positional Encoding
        x_seq = x_seq + self.pos_encoder[:, :D, :] # Use only up to D slices/elements
        x_seq = self.input_projection(x_seq) 
        
        # 4. Directional Attention (Sequence Attention)
        z_out = self.transformer_encoder(x_seq) 
        
        # 5. Modulation Vector Projection
        z_modulation = self.output_projection(z_out) # Output: (N, Seq_len, 32)
        
        return z_modulation


# ------------------------------
# 3D FCBFormer (The main model)
# ------------------------------
class FCBFormer(nn.Module):
    """The combined 3D CNN and 3-Directional Transformer model."""
    def __init__(self, in_channels, out_channels, size): 
        super().__init__()
        
        # FCB_OUT_CHANNELS determines the channel count for the U-Net bottleneck/fusion (32)
        FCB_OUT_CHANNELS = 32
        
        # --- 1. Fully Convolutional Branch ---
        self.FCB = FCB(in_channels=in_channels, min_level_channels=FCB_OUT_CHANNELS) 
        
        # --- 2. Directional Transformer Branches (Z, Y, X) ---
        # The 'size' (128) is the sequence length for all axes since D=H=W=128
        self.TB_Z = TB(in_channels=in_channels, seq_len=size, fc_channels=FCB_OUT_CHANNELS) 
        self.TB_Y = TB(in_channels=in_channels, seq_len=size, fc_channels=FCB_OUT_CHANNELS) 
        self.TB_X = TB(in_channels=in_channels, seq_len=size, fc_channels=FCB_OUT_CHANNELS) 
        
        # --- 3. Prediction Heads (One for each directional fusion) ---
        def create_ph():
            return nn.Sequential(
                RB(FCB_OUT_CHANNELS, FCB_OUT_CHANNELS),
                RB(FCB_OUT_CHANNELS, FCB_OUT_CHANNELS),
                # Use out_channels (1) for the final convolution to output the dose map
                nn.Conv3d(FCB_OUT_CHANNELS, out_channels, kernel_size=1) 
            )

        self.PH_Z = create_ph()
        self.PH_Y = create_ph()
        self.PH_X = create_ph()

    def forward(self, x):
        # Input x: (N, C_in, D, H, W) e.g., (N, 3, 128, 128, 128)
        
        # --- 1. FCB Path (3D CNN) ---
        x_FCB = self.FCB(x) # Output: (N, C_fcb, D, H, W) e.g., (N, 32, 128, 128, 128)
        C_fcb = x_FCB.shape[1]

        # --- 2. Z-Axis Fusion (Sequence along D, spatial along H, W) ---
        x_TB_Z = self.TB_Z(x) # Output: (N, D, C_fcb) e.g., (N, 128, 32)
        # Reshape Z-Modulator: (N, D, C_fcb) -> (N, C_fcb, D, 1, 1)
        z_mod_reshaped_Z = x_TB_Z.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        x_fused_Z = x_FCB * z_mod_reshaped_Z 
        out_Z = self.PH_Z(x_fused_Z) 

        # --- 3. Y-Axis Fusion (Sequence along H, spatial along D, W) ---
        # Permute input for TB: (N, C, D, H, W) -> (N, C, H, D, W)
        x_Y = x.permute(0, 1, 3, 2, 4) 
        x_TB_Y = self.TB_Y(x_Y) # Output: (N, H, C_fcb) e.g., (N, 128, 32)
        # Reshape Y-Modulator: (N, H, C_fcb) -> (N, C_fcb, H, 1, 1)
        z_mod_reshaped_Y = x_TB_Y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        
        # Permute FCB output to match Y-axis sequence: (N, C, D, H, W) -> (N, C, H, D, W)
        x_FCB_Y = x_FCB.permute(0, 1, 3, 2, 4) 
        x_fused_Y = x_FCB_Y * z_mod_reshaped_Y
        
        # Permute fused back to original orientation: (N, C, H, D, W) -> (N, C, D, H, W)
        x_fused_Y_orig = x_fused_Y.permute(0, 1, 3, 2, 4) 
        out_Y = self.PH_Y(x_fused_Y_orig)

        # --- 4. X-Axis Fusion (Sequence along W, spatial along D, H) ---
        # Permute input for TB: (N, C, D, H, W) -> (N, C, W, D, H)
        x_X = x.permute(0, 1, 4, 2, 3)
        x_TB_X = self.TB_X(x_X) # Output: (N, W, C_fcb) e.g., (N, 128, 32)
        # Reshape X-Modulator: (N, W, C_fcb) -> (N, C_fcb, W, 1, 1)
        z_mod_reshaped_X = x_TB_X.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        # Permute FCB output to match X-axis sequence: (N, C, D, H, W) -> (N, C, W, D, H)
        x_FCB_X = x_FCB.permute(0, 1, 4, 2, 3) 
        x_fused_X = x_FCB_X * z_mod_reshaped_X
        
        # Permute fused back to original orientation: (N, C, W, D, H) -> (N, C, D, H, W)
        x_fused_X_orig = x_fused_X.permute(0, 1, 3, 4, 2) 
        out_X = self.PH_X(x_fused_X_orig)
        
        # --- 5. Final Output (Mean of the three directions) ---
        out = (out_Z + out_Y + out_X) / 3.0 
        
        return out
