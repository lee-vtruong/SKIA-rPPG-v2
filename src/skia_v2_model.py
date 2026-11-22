"""SKIA-Net v2: Temporally Stable Anatomical Region Gating for Motion-Robust rPPG."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from roi_extractor import make_geometric_region_masks


def standardize_waveform(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean, unit-std temporal normalization preventing amplitude collapse."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-invariant negative Pearson correlation loss over temporal axis."""
    p = standardize_waveform(pred.float(), eps=eps)
    t = standardize_waveform(target.float(), eps=eps)
    corr = (p * t).mean(dim=-1)
    return (1.0 - corr).mean()


def spectral_psd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    nfft: int = 512,
) -> torch.Tensor:
    """
    Differentiable spectral distribution alignment (KL divergence) within [0.7, 2.5] Hz.
    """
    B, T = pred.shape
    p = standardize_waveform(pred.float())
    t = standardize_waveform(target.float())

    win = torch.hann_window(T, device=pred.device, dtype=torch.float32).unsqueeze(0)
    nfft_eff = max(nfft, T)
    freqs = torch.fft.rfftfreq(nfft_eff, d=1.0 / fs, device=pred.device)
    band = (freqs >= low_hz) & (freqs <= high_hz)

    p_fft = torch.fft.rfft(p * win, n=nfft_eff)
    t_fft = torch.fft.rfft(t * win, n=nfft_eff)

    p_pow = (p_fft.real ** 2 + p_fft.imag ** 2)[:, band] + 1e-8
    t_pow = (t_fft.real ** 2 + t_fft.imag ** 2)[:, band] + 1e-8

    p_prob = p_pow / p_pow.sum(dim=-1, keepdim=True)
    t_prob = t_pow / t_pow.sum(dim=-1, keepdim=True)

    return F.kl_div(torch.log(p_prob), t_prob, reduction="batchmean")


def temporal_smoothness_loss(weights: torch.Tensor) -> torch.Tensor:
    """L1 temporal smoothness regularization: (1/(T-1)) * sum ||w_t - w_{t-1}||_1."""
    if weights.shape[-1] < 2:
        return weights.new_zeros((), dtype=torch.float32)
    w = weights.float()
    diff = torch.abs(w[:, :, 1:] - w[:, :, :-1])
    return diff.sum(dim=1).mean()


class SKIAv2Loss(nn.Module):
    """Combined time-domain Pearson + Spectral PSD + Region-Gate Smoothness objective."""

    def __init__(
        self,
        lambda_psd: float = 0.2,
        lambda_smooth: float = 0.1,
        fs: float = 30.0,
    ):
        super().__init__()
        self.lambda_psd = lambda_psd
        self.lambda_smooth = lambda_smooth
        self.fs = fs

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_pearson = neg_pearson_loss(pred, target)
        l_psd = spectral_psd_loss(pred, target, fs=self.fs) if self.lambda_psd > 0 else pred.new_zeros(())
        l_smooth = (
            temporal_smoothness_loss(weights)
            if (weights is not None and self.lambda_smooth > 0)
            else pred.new_zeros(())
        )
        total = l_pearson + self.lambda_psd * l_psd + self.lambda_smooth * l_smooth
        stats = {
            "loss_total": float(total.detach().item()),
            "loss_pearson": float(l_pearson.detach().item()),
            "loss_psd": float(l_psd.detach().item()),
            "loss_smooth": float(l_smooth.detach().item()),
        }
        return total, stats


class TemporallyStableRegionGate(nn.Module):
    """
    Computes temporally smooth reliability weights w_{k,t} across K anatomical facial ROIs
    from localized motion energy and regional spatiotemporal features.
    """

    def __init__(
        self,
        num_regions: int = 4,
        feat_dim: int = 64,
        gate_mode: str = "stable_dynamic",
        smooth_kernel: int = 11,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.gate_mode = gate_mode
        self.smooth_kernel = smooth_kernel

        # Anatomical prior: Forehead (0), L-Cheek (1), R-Cheek (2) > Mouth/Jaw (3)
        init_bias = torch.tensor([0.6, 0.5, 0.5, -0.8][:num_regions], dtype=torch.float32)
        if len(init_bias) < num_regions:
            init_bias = torch.zeros(num_regions, dtype=torch.float32)
        self.anat_bias = nn.Parameter(init_bias)
        self.motion_gamma = nn.Parameter(torch.ones(num_regions) * 1.5)

        self.motion_proj = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=1),
        )
        self.feat_proj = nn.Sequential(
            nn.Conv1d(feat_dim, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

        # Fixed Gaussian smoothing kernel buffer for temporal stability
        half = smooth_kernel // 2
        grid = torch.arange(-half, half + 1, dtype=torch.float32)
        sigma = max(1.0, smooth_kernel / 4.0)
        gauss = torch.exp(-0.5 * (grid / sigma) ** 2)
        gauss = gauss / gauss.sum()
        self.register_buffer("smooth_filter", gauss.view(1, 1, -1))

    def forward(
        self,
        region_feats: torch.Tensor,
        region_motion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          region_feats: (B, K, C, T)
          region_motion: (B, K, T)
        Returns:
          weights: (B, K, T) summing to 1 along K
        """
        B, K, C, T = region_feats.shape

        if self.gate_mode == "static_prior":
            logits = self.anat_bias.view(1, K, 1).expand(B, K, T)
            return F.softmax(logits, dim=1)

        # Normalize motion energy per clip across regions & time for scale invariance
        m_mean = region_motion.mean(dim=(1, 2), keepdim=True)
        m_std = region_motion.std(dim=(1, 2), keepdim=True) + 1e-6
        m_norm = (region_motion - m_mean) / m_std  # (B, K, T)

        m_flat = m_norm.reshape(B * K, 1, T)
        m_score = self.motion_proj(m_flat).reshape(B, K, T)

        f_flat = region_feats.reshape(B * K, C, T)
        f_score = self.feat_proj(f_flat).reshape(B, K, T)

        gamma = F.softplus(self.motion_gamma).view(1, K, 1)
        logits = self.anat_bias.view(1, K, 1) - gamma * torch.sigmoid(m_score) + 0.5 * torch.tanh(f_score)

        if self.gate_mode == "stable_dynamic" and T >= self.smooth_kernel:
            pad = self.smooth_kernel // 2
            l_flat = logits.reshape(B * K, 1, T)
            l_pad = F.pad(l_flat, (pad, pad), mode="replicate")
            logits = F.conv1d(l_pad, self.smooth_filter).reshape(B, K, T)

        return F.softmax(logits, dim=1)


class SKIANetV2(nn.Module):
    """
    SKIA-Net v2: Anatomy-Aware, Temporally Stable Region Gating Network for rPPG.

    Uses complementary appearance X and temporal-difference dX representations,
    extracts localized motion energy and features over K anatomical ROIs
    (forehead, left cheek, right cheek, mouth/jaw), gates regions with temporal
    smoothness, and outputs a variance-stabilized BVP waveform.
    """

    def __init__(
        self,
        num_regions: int = 4,
        base_channels: int = 16,
        gate_mode: str = "stable_dynamic",
        use_diff: bool = True,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.gate_mode = gate_mode
        self.use_diff = use_diff
        in_ch = 6 if use_diff else 3

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4  # 64

        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, c1, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(c1),
            nn.ELU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(c1, c2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(c2),
            nn.ELU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(c2, c3, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(c3),
            nn.ELU(inplace=True),
        )

        self.region_gate = TemporallyStableRegionGate(
            num_regions=num_regions,
            feat_dim=c3,
            gate_mode=gate_mode,
        )

        self.global_alpha = nn.Parameter(torch.tensor(0.25))

        self.temporal_head = nn.Sequential(
            nn.Conv1d(c3, c3, kernel_size=5, padding=2),
            nn.ELU(inplace=True),
            nn.Conv1d(c3, c2, kernel_size=5, padding=2),
            nn.ELU(inplace=True),
            nn.Conv1d(c2, c2, kernel_size=5, padding=4, dilation=2),
            nn.ELU(inplace=True),
            nn.Conv1d(c2, 1, kernel_size=1),
        )

        self._cached_default_mask: Optional[torch.Tensor] = None

    def _default_masks(self, B: int, T: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._cached_default_mask is None
            or self._cached_default_mask.shape[-2:] != (H, W)
            or self._cached_default_mask.device != device
        ):
            geom = make_geometric_region_masks(H, W)[: self.num_regions]
            self._cached_default_mask = torch.from_numpy(geom).to(device=device, dtype=dtype).view(1, self.num_regions, 1, H, W)
        return self._cached_default_mask.expand(B, self.num_regions, T, H, W)

    def forward(
        self,
        x: torch.Tensor,
        region_masks: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          x: (B, 3, T, H, W) normalized face-crop clip, or (B, >=6, T, H, W) if diff concatenated
          region_masks: optional (B, K, T, H, W) anatomical masks in [0, 1]
        Returns:
          bvp: (B, T) unit-variance reconstructed BVP waveform
          (optional) weights: (B, K, T) region attention weights
        """
        if x.shape[1] > 3:
            rgb = x[:, :3]
        else:
            rgb = x

        B, _, T, H, W = rgb.shape

        diff = torch.zeros_like(rgb)
        if T > 1:
            diff[:, :, 1:] = rgb[:, :, 1:] - rgb[:, :, :-1]

        enc_in = torch.cat([rgb, diff], dim=1) if self.use_diff else rgb
        feat = self.encoder(enc_in)  # (B, C3, T, H', W')
        _, C3, _, Hp, Wp = feat.shape

        if region_masks is None:
            masks = self._default_masks(B, T, H, W, rgb.device, rgb.dtype)
        else:
            masks = region_masks[:, : self.num_regions]

        # Non-parametric mask downsampling & motion stats need no autograd graph over model weights
        with torch.no_grad():
            if self.gate_mode == "static_prior":
                region_motion = torch.zeros(B, self.num_regions, T, device=rgb.device, dtype=rgb.dtype)
            else:
                diff_mag = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-8)  # (B, 1, T, H, W)
                mask_sum_hw = masks.sum(dim=(-2, -1)) + 1e-6  # (B, K, T)
                region_motion = (masks * diff_mag).sum(dim=(-2, -1)) / mask_sum_hw  # (B, K, T)

            # Downsample anatomical masks to spatial resolution (H', W') of encoder features
            masks_ds = F.adaptive_avg_pool3d(masks, output_size=(T, Hp, Wp))  # (B, K, T, H', W')
            ds_sum = masks_ds.sum(dim=(-2, -1)).unsqueeze(2) + 1e-6  # (B, K, 1, T)

        # Regional pooled features: (B, K, C3, T)
        region_feats = torch.einsum("bcthw,bkthw->bkct", feat, masks_ds) / ds_sum
        global_feat = feat.mean(dim=(-2, -1))  # (B, C3, T)

        weights = self.region_gate(region_feats, region_motion)  # (B, K, T)
        fused_region = torch.einsum("bkt,bkct->bct", weights, region_feats)
        fused = fused_region + self.global_alpha * global_feat

        raw_out = self.temporal_head(fused).squeeze(1)  # (B, T)
        out = standardize_waveform(raw_out)

        if return_weights:
            return out, weights
        return out
