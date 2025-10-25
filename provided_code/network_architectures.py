# provided_code_torch/network_architectures.py
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_padding(kernel: Tuple[int, int, int], stride: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Compute numeric padding compatible with stride>=1 (avoids padding='same').
    For common (k=4, s=2) -> 1; for stride=1 use k//2.
    """
    pads = []
    for k, s in zip(kernel, stride):
        if s == 1:
            pads.append(k // 2)  # classic "same-like" for odd k
        else:
            pads.append(max((k - s + 1) // 2, 1))
    return tuple(pads)


def conv_block(in_ch: int, out_ch: int, kernel=(4, 4, 4), stride=(2, 2, 2), use_bn: bool = True):
    padding = _calc_padding(kernel, stride)
    layers = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False)]
    if use_bn:
        layers.append(nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.99))
    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*layers)


class ConvTBlock(nn.Module):
    """
    ConvTranspose3d -> BN -> (optional Dropout3d) -> ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, kernel=(4, 4, 4), stride=(2, 2, 2), dropout_p: float = 0.0):
        super().__init__()
        padding = _calc_padding(kernel, stride)
        self.convT = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, output_padding=0, bias=False
        )
        self.bn = nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.99)
        self.do = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convT(x)
        x = self.bn(x)
        x = self.do(x)
        x = self.act(x)
        return x


class DoseFromCT3DUNet(nn.Module):
    """
    PyTorch reimplementation of the Keras generator used in OpenKBP starter code.

    Accepts channels-first (N,C,D,H,W) or channels-last (N,D,H,W,C) inputs and
    converts to channels-first internally.

    Inputs:
      - ct:        (N, 1, D, H, W)  or (N, D, H, W, 1)
      - roi_masks: (N, M, D, H, W)  or (N, D, H, W, M)
    Output:
      - dose:      (N, 1, D, H, W)
    """
    def __init__(
        self,
        ct_channels: int = 1,
        mask_channels: int = 1,
        initial_filters: int = 16,               # consider 64+ for real training
        kernel: Tuple[int, int, int] = (4, 4, 4),
        stride: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.kernel = kernel
        self.stride = stride

        in_ch = ct_channels + mask_channels

        # Encoder
        self.conv1 = conv_block(in_ch,               initial_filters,               kernel, stride, use_bn=True)
        self.conv2 = conv_block(initial_filters,     2 * initial_filters,           kernel, stride, use_bn=True)
        self.conv3 = conv_block(2 * initial_filters, 4 * initial_filters,           kernel, stride, use_bn=True)
        self.conv4 = conv_block(4 * initial_filters, 8 * initial_filters,           kernel, stride, use_bn=True)
        self.conv5 = conv_block(8 * initial_filters, 8 * initial_filters,           kernel, stride, use_bn=True)
        self.conv6 = conv_block(8 * initial_filters, 8 * initial_filters,           kernel, stride, use_bn=False)  # bottleneck

        # Decoder
        self.up5 = ConvTBlock(8 * initial_filters,  8 * initial_filters, kernel, stride, dropout_p=0.0)
        self.up4 = ConvTBlock(16 * initial_filters, 8 * initial_filters, kernel, stride, dropout_p=0.2)
        self.up3 = ConvTBlock(16 * initial_filters, 4 * initial_filters, kernel, stride, dropout_p=0.0)
        self.up2 = ConvTBlock(8 * initial_filters,  2 * initial_filters, kernel, stride, dropout_p=0.2)
        self.up1 = ConvTBlock(4 * initial_filters,  initial_filters,     kernel, stride, dropout_p=0.0)

        # Final head
        self.final_convT = nn.ConvTranspose3d(
            2 * initial_filters, 1, kernel_size=kernel, stride=stride, padding=_calc_padding(kernel, stride)
        )
        self.avgpool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _to_channels_first(ct: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Robustly convert to channels-first:
          ct:    (N,1,D,H,W) or (N,D,H,W,1) -> (N,1,D,H,W)
          masks: (N,M,D,H,W) or (N,D,H,W,M) -> (N,M,D,H,W)
        Heuristic for masks: if the last dim is "small" (<=128) and the second dim
        looks like a spatial dimension (>> channels), treat as channels-last.
        """
        # CT: channels-last if last dim == 1 and second dim looks spatial
        if ct.ndim == 5 and ct.shape[1] != 1 and ct.shape[-1] == 1:
            ct = ct.permute(0, 4, 1, 2, 3).contiguous()

        # Masks: channels-last if last dim is small (typical #ROIs) and dim1 looks spatial
        if masks.ndim == 5:
            likely_channels_last = (masks.shape[-1] <= 128) and (masks.shape[1] >= 16)
            if likely_channels_last:
                masks = masks.permute(0, 4, 1, 2, 3).contiguous()

        return ct, masks

    def forward(self, ct: torch.Tensor, roi_masks: torch.Tensor):
        # Ensure channels-first
        ct, roi_masks = self._to_channels_first(ct, roi_masks)

        # Align spatial grids (upsample masks to CT grid if needed)
        if roi_masks.shape[2:] != ct.shape[2:]:
            roi_masks = F.interpolate(roi_masks, size=ct.shape[2:], mode="nearest")

        # Concatenate inputs
        x = torch.cat([ct, roi_masks], dim=1)  # (N, 1+M, D, H, W)

        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # Decoder with skips
        y5 = self.up5(x6)
        y4 = self.up4(torch.cat([y5, x5], dim=1))
        y3 = self.up3(torch.cat([y4, x4], dim=1))
        y2 = self.up2(torch.cat([y3, x3], dim=1))
        y1 = self.up1(torch.cat([y2, x2], dim=1))

        y0 = torch.cat([y1, x1], dim=1)
        y0 = self.final_convT(y0)
        y0 = self.avgpool(y0)
        out = self.relu(y0)
        return out
