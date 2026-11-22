"""Standardized deep rPPG baselines: PhysNet, TS-CAN, EfficientPhys, PhysFormer, and SKIA-v1."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from skia_model import SKIANetLite
from skia_v2_model import standardize_waveform


class TemporalShift(nn.Module):
    """Temporal Shift Module (TSM) along temporal axis T for 5D (B, C, T, H, W) tensors."""

    def __init__(self, n_div: int = 3):
        super().__init__()
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if T <= 1:
            return x
        fold = max(1, C // self.n_div)
        out = torch.zeros_like(x)
        out[:, :fold, :-1] = x[:, :fold, 1:]
        out[:, fold:2 * fold, 1:] = x[:, fold:2 * fold, :-1]
        out[:, 2 * fold:] = x[:, 2 * fold:]
        return out


class PhysNet(nn.Module):
    """3D CNN Encoder-Decoder PhysNet (Yu et al., BMVC 2019) preserving temporal length T."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ELU(inplace=True),
        )
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
        )
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
        )
        self.head = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_in = x.shape[2]
        h = self.conv1(x)
        h = self.loop1(h)
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.decoder1(h)
        h = self.decoder2(h)
        h = h.mean(dim=(-2, -1), keepdim=True)  # (B, 32, T', 1, 1)
        out = self.head(h).squeeze(-1).squeeze(-1).squeeze(1)  # (B, T')
        if out.shape[-1] != T_in:
            out = F.interpolate(out.unsqueeze(1), size=T_in, mode="linear", align_corners=False).squeeze(1)
        return standardize_waveform(out)


def _resize_video_spatial(x: torch.Tensor, target_hw: int = 72) -> torch.Tensor:
    """Downsample 5D (B, C, T, H, W) tensor spatially to match 2D TSM rPPG-Toolbox resolution."""
    B, C, T, H, W = x.shape
    if H <= target_hw and W <= target_hw:
        return x
    x2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    x2d = F.interpolate(x2d, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return x2d.reshape(B, T, C, target_hw, target_hw).permute(0, 2, 1, 3, 4)


class TSCAN(nn.Module):
    """
    Temporal Shift Convolutional Attention Network (Liu et al., NeurIPS 2020).
    Combines Appearance attention branch with Temporal-Shift Motion branch.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.tsm = TemporalShift(n_div=3)

        # Appearance branch
        self.app_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        self.attn1 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid())
        self.app_pool1 = nn.AvgPool2d(2)

        self.app_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.attn2 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

        # Motion branch
        self.mot_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )
        self.mot_pool1 = nn.AvgPool2d(2)
        self.mot_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )

        self.head = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_video_spatial(x, target_hw=72)
        B, C, T, H, W = x.shape
        diff = torch.zeros_like(x)
        if T > 1:
            diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

        # Collapse (B, T) into batch for 2D spatial ops
        app_in = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        a1 = self.app_conv1(app_in)
        m1_mask = self.attn1(a1)
        m1_mask = m1_mask / (m1_mask.mean(dim=(-2, -1), keepdim=True) + 1e-6) * 0.5

        mot_t = self.tsm(diff).permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        m1 = self.mot_conv1(mot_t) * m1_mask

        a1_p = self.app_pool1(a1)
        m1_p = self.mot_pool1(m1)

        a2 = self.app_conv2(a1_p)
        m2_mask = self.attn2(a2)
        m2_mask = m2_mask / (m2_mask.mean(dim=(-2, -1), keepdim=True) + 1e-6) * 0.5

        # Reshape back for second temporal shift
        m1_5d = m1_p.reshape(B, T, 16, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        m1_shifted = self.tsm(m1_5d).permute(0, 2, 1, 3, 4).reshape(B * T, 16, H // 2, W // 2)
        m2 = self.mot_conv2(m1_shifted) * m2_mask

        pooled = m2.mean(dim=(-2, -1)).reshape(B, T, 32).permute(0, 2, 1)  # (B, 32, T)
        out = self.head(pooled).squeeze(1)
        return standardize_waveform(out)


class EfficientPhys(nn.Module):
    """
    EfficientPhys (Liu et al., WACV 2023): Batch-normalized temporal differences,
    Temporal Shift Modules, and lightweight spatial self-attention.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.diff_bn = nn.BatchNorm2d(in_channels)
        self.tsm = TemporalShift(n_div=3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.pool2 = nn.AvgPool2d(2)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_video_spatial(x, target_hw=72)
        B, C, T, H, W = x.shape
        diff = torch.zeros_like(x)
        if T > 1:
            diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

        d2d = self.diff_bn(diff.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
        d5d = d2d.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        h = self.conv1(self.tsm(d5d).permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))
        h = self.pool1(h)

        h5d = h.reshape(B, T, 16, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        h = self.conv2(self.tsm(h5d).permute(0, 2, 1, 3, 4).reshape(B * T, 16, H // 2, W // 2))
        h = self.pool2(h)

        attn = self.spatial_attn(h)
        attn = attn / (attn.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        feat = (h * attn).sum(dim=(-2, -1)).reshape(B, T, 32).permute(0, 2, 1)  # (B, 32, T)
        out = self.head(feat).squeeze(1)
        return standardize_waveform(out)


class PhysFormer(nn.Module):
    """
    Lightweight Temporal-Difference Transformer for rPPG (Yu et al., CVPR 2022).
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 4, 4)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 4, 4)),
            nn.Conv3d(32, embed_dim, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Temporal Difference Self-Attention over spatiotemporal tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        # Incorporate temporal difference stem cue
        diff = torch.zeros_like(x)
        if T > 1:
            diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        feat = self.stem(x + 0.5 * diff)  # (B, D, T, H', W')
        # Pool spatially to a compact 2x2 grid per frame so T*4 tokens fit fast on 6GB GPU
        feat_grid = F.adaptive_avg_pool3d(feat, output_size=(T, 2, 2))  # (B, D, T, 2, 2)
        D = feat_grid.shape[1]
        tokens = feat_grid.permute(0, 2, 3, 4, 1).reshape(B, T * 4, D)
        tokens = self.transformer(tokens)
        seq = tokens.reshape(B, T, 4, D).mean(dim=2).permute(0, 2, 1)  # (B, D, T)
        out = self.head(seq).squeeze(1)
        return standardize_waveform(out)


class SKIAV1Wrapper(nn.Module):
    """
    Wrapper around fixed SKIANetLite accepting standardized (B, 3, T, H, W) clips
    so SKIA-v1 variants can be benchmarked directly alongside PhysNet, TS-CAN, EfficientPhys, PhysFormer.
    """

    def __init__(self, mode: str = "full", ratio: float = 0.7):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        in_ch = {"raw": 3, "masked": 3, "raw_masked": 6, "full": 9}[mode]
        self.net = SKIANetLite(in_channels=in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        mask = torch.ones((1, 1, 1, H, W), device=x.device, dtype=x.dtype)
        mask[:, :, :, int(H * self.ratio):, :] = 0.0
        masked = x * mask
        if self.mode == "raw":
            inp = x
        elif self.mode == "masked":
            inp = masked
        elif self.mode == "raw_masked":
            inp = torch.cat([x, masked], dim=1)
        elif self.mode == "full":
            diff = torch.zeros_like(x)
            if T > 1:
                diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            inp = torch.cat([x, masked, diff], dim=1)
        else:
            raise ValueError(self.mode)
        return self.net(inp)


def build_deep_model(name: str, **kwargs) -> nn.Module:
    """Factory mapping standardized model names to nn.Module instances."""
    key = name.lower()
    if key == "physnet":
        return PhysNet(**kwargs)
    if key in ("tscan", "ts-can", "ts_can"):
        return TSCAN(**kwargs)
    if key == "efficientphys":
        return EfficientPhys(**kwargs)
    if key == "physformer":
        return PhysFormer(**kwargs)
    if key == "skia_raw":
        return SKIAV1Wrapper(mode="raw")
    if key == "skia_masked":
        return SKIAV1Wrapper(mode="masked")
    if key == "skia_raw_masked":
        return SKIAV1Wrapper(mode="raw_masked")
    if key in ("skia_full", "skia_v1"):
        return SKIAV1Wrapper(mode="full")
    raise ValueError(f"Unknown baseline model: {name}")
