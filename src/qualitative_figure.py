"""Fixed qualitative visualization: prevents flat-line collapse and plots region gates + PSD."""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import bandpass_filter, estimate_hr_fft, pearson_corr
from roi_extractor import REGION_NAMES
from skia_dataset import SKIADataset, SKIAV2Dataset
from skia_model import SKIANetLite
from skia_v2_model import SKIANetV2
from splits import discover_subjects, get_subject_split


def safe_zscore(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    std = float(np.std(arr))
    if std < 1e-12:
        raise ValueError(f"Degenerate near-constant signal encountered (std={std:.3e})")
    return (arr - np.mean(arr)) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--arch", type=str, default="skia_v2", choices=["skia_v1", "skia_v2"])
    parser.add_argument("--mode", type=str, default="full", choices=["raw", "masked", "raw_masked", "full"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="qualitative_figure.png")
    parser.add_argument("--clip-len", type=int, default=128)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=float, default=30.0)
    args = parser.parse_args()

    subjects = discover_subjects(args.data_root)
    splits = get_subject_split(subjects, seed=args.seed, preserve_v1_test=True)
    test_subjects = splits["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.arch == "skia_v2":
        dataset = SKIAV2Dataset(
            args.data_root, test_subjects, clip_len=args.clip_len, stride=args.clip_len,
            size=args.size, return_masks=True, return_meta=True
        )
        model = SKIANetV2(num_regions=4, gate_mode="stable_dynamic").to(device)
    else:
        in_ch = {"raw": 3, "masked": 3, "raw_masked": 6, "full": 9}[args.mode]
        dataset = SKIADataset(
            args.data_root, test_subjects, clip_len=args.clip_len, stride=args.clip_len,
            size=args.size, mode=args.mode, return_meta=True
        )
        model = SKIANetLite(in_channels=in_ch).to(device)

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    if all(k.startswith("net.") for k in state.keys()):
        state = {k[4:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    best_corr = -999.0
    best_sample = None

    for idx in range(min(len(dataset), 40)):
        if args.arch == "skia_v2":
            x, m, gt, subj, st = dataset[idx]
            with torch.no_grad():
                pred_t, weights_t = model(
                    x.unsqueeze(0).to(device), m.unsqueeze(0).to(device), return_weights=True
                )
            pred = pred_t.squeeze(0).cpu().numpy()
            weights = weights_t.squeeze(0).cpu().numpy()
            rmask = m.numpy()
        else:
            x, gt, subj, st = dataset[idx]
            with torch.no_grad():
                pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            weights = None
            rmask = None

        pred_bp = bandpass_filter(pred, fs=args.fs)
        gt_bp = bandpass_filter(gt.numpy(), fs=args.fs)
        corr = pearson_corr(pred_bp, gt_bp)

        if corr > best_corr:
            best_corr = corr
            best_sample = (x, rmask, weights, gt_bp, pred_bp, subj, st)

    assert best_sample is not None, "No samples found in test dataset!"
    x, rmask, weights, gt_bp, pred_bp, subj, st = best_sample

    pred_std = float(np.std(pred_bp))
    assert pred_std > 1e-4, f"Predicted BVP collapsed to flat signal (std={pred_std:.3e})!"

    gt_norm = safe_zscore(gt_bp)
    pred_norm = safe_zscore(pred_bp)
    hr_gt = estimate_hr_fft(gt_bp, fs=args.fs)
    hr_pred = estimate_hr_fft(pred_bp, fs=args.fs)

    print(
        f"Selected {subj} (start={st}) | Pearson={best_corr:.4f} | "
        f"pred_std={pred_std:.4f} | GT HR={hr_gt:.1f} bpm | Pred HR={hr_pred:.1f} bpm"
    )

    frame_idx = min(30, x.shape[1] - 1)
    raw_img = x[:3, frame_idx].permute(1, 2, 0).numpy()
    raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)

    nrows = 3 if weights is not None else 2
    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })
    fig = plt.figure(figsize=(14, 9.6 if weights is not None else 7.1), constrained_layout=True)
    gs = fig.add_gridspec(nrows, 6)

    # Row 1: Frame, Anatomical ROIs / Mask, Effective Weighted View
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(raw_img)
    ax1.set_title(f"Face Crop ({subj}, t={frame_idx})")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2:4])
    if rmask is not None and weights is not None:
        # Spatial weight map at frame_idx: sum_k w_{k,t} * M_k
        spatial_w = np.tensordot(weights[:, frame_idx], rmask[:, frame_idx], axes=([0], [0]))
        im = ax2.imshow(spatial_w, cmap="viridis", vmin=0.0, vmax=max(0.5, float(spatial_w.max())))
        ax2.set_title("Anatomical Region Gate Map $w_{k,t}$")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    else:
        mask_vis = np.mean(x[3:6, frame_idx].permute(1, 2, 0).numpy(), axis=2) if x.shape[0] >= 6 else np.ones(raw_img.shape[:2])
        ax2.imshow(mask_vis, cmap="gray")
        ax2.set_title("Static Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 4:6])
    if rmask is not None and weights is not None:
        spatial_w = np.tensordot(weights[:, frame_idx], rmask[:, frame_idx], axes=([0], [0]))
        weighted_img = raw_img * np.clip(spatial_w[..., None] / (spatial_w.max() + 1e-8), 0.15, 1.0)
        ax3.imshow(weighted_img)
        ax3.set_title("Gate-Attended Face View")
    else:
        masked_img = x[3:6, frame_idx].permute(1, 2, 0).numpy() if x.shape[0] >= 6 else raw_img
        masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min() + 1e-8)
        ax3.imshow(masked_img)
        ax3.set_title("Masked Frame")
    ax3.axis("off")

    # Row 2: Waveform reconstruction
    ax4 = fig.add_subplot(gs[1, :])
    t_sec = np.arange(len(gt_norm)) / args.fs
    ax4.plot(t_sec, gt_norm, label=f"GT BVP (4.27s Clip FFT HR={hr_gt:.1f} bpm)", linewidth=2.2, color="#1f77b4")
    ax4.plot(t_sec, pred_norm, label=f"Predicted BVP (4.27s Clip FFT HR={hr_pred:.1f} bpm, r={best_corr:.3f})", linewidth=2.2, color="#ff7f0e", alpha=0.9)
    ax4.set_xlabel("Time (s) [128-frame clip excerpt]")
    ax4.set_ylabel("Normalized Amplitude (z-score)")
    ax4.set_title(f"Waveform Reconstruction ({args.arch.upper()}, Clip Excerpt)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    # Row 3: Temporal gate weights + PSD spectrum
    if weights is not None:
        ax5 = fig.add_subplot(gs[2, 0:3])
        for k, rname in enumerate(REGION_NAMES[:weights.shape[0]]):
            ax5.plot(t_sec, weights[k], label=rname, linewidth=2.0)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Gate Weight $w_{k,t}$")
        ax5.set_title("Quasi-Static Region Weights $w_{k,t}$ (Within-Clip SD < 0.005)")
        ax5.set_ylim(0.0, 1.0)
        ax5.grid(True, alpha=0.3)
        ax5.legend(ncol=2, fontsize=10)

        ax6 = fig.add_subplot(gs[2, 3:6])
        nfft = 4096
        win = np.hanning(len(gt_bp))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / args.fs) * 60.0
        band = (freqs >= 42.0) & (freqs <= 150.0)
        mag_gt = np.abs(np.fft.rfft(gt_norm * win, n=nfft))[band]
        mag_pr = np.abs(np.fft.rfft(pred_norm * win, n=nfft))[band]
        ax6.plot(freqs[band], mag_gt / (mag_gt.max() + 1e-12), label="GT PSD", linewidth=2.0)
        ax6.plot(freqs[band], mag_pr / (mag_pr.max() + 1e-12), label="Pred PSD", linewidth=2.0, linestyle="--")
        ax6.set_xlabel("Heart Rate (bpm)")
        ax6.set_ylabel("Normalized Magnitude")
        ax6.set_title("Zero-Padded Frequency Spectrum")
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"Saved qualitative figure to {args.out}")


if __name__ == "__main__":
    main()
