"""Generate publication-quality figures for the SKIA-v2 ACIIDS 2027 manuscript directly from verified CSVs."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


RES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "ubfc_full"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "paper_aciids2027", "figures"))


def make_architecture_diagram(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.4), dpi=300)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#F0F5FA", ec="#2B5B84", lw=1.5, title_fs=9.2, sub_fs=7.8):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        if sub:
            ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color="#102A43")
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                    fontsize=sub_fs, color="#334E68", linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color="#102A43")

    def arrow(x1, y1, x2, y2, color="#2B5B84", style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6, shrinkA=2, shrinkB=2))

    # Header band labels
    ax.text(1.40, 4.12, "1. Landmark ROI Extraction", ha="center", fontsize=9.3, fontweight="bold", color="#1F4E78")
    ax.text(4.35, 4.12, "2. Spatiotemporal Encoding", ha="center", fontsize=9.3, fontweight="bold", color="#1F4E78")
    ax.text(7.40, 4.12, "3. Temporally Stable Gating", ha="center", fontsize=9.3, fontweight="bold", color="#1F4E78")
    ax.text(10.45, 4.12, "4. Pulse Head & Joint Loss", ha="center", fontsize=9.3, fontweight="bold", color="#1F4E78")

    # Stage 1
    box(0.20, 2.38, 2.40, 1.25, "Facial Video Clip $X$",
        "RGB $(B, 3, T, H, W)$\nEMA-Stabilized Crop\n($\\alpha_{\\mathrm{EMA}}=0.85, 128\\times 128$)",
        fc="#EBF4FF", ec="#1D4ED8")
    box(0.20, 0.45, 2.40, 1.35, "Anatomical Masks $M$",
        "$K=4$ Convex Hulls:\nForehead, L/R Cheek,\nMouth/Jaw (478/68-pt)",
        fc="#ECFDF5", ec="#047857")

    # Stage 2
    box(3.15, 2.42, 2.40, 1.20, "3D-CNN Encoder",
        "Input: $[X; \\Delta X]$ (6 ch)\n$16\\rightarrow 32\\rightarrow 64$ ch, AvgPool3d\nFeat $F\\;(B, C, T, H', W')$",
        fc="#EEF2FF", ec="#4338CA")
    box(3.15, 0.50, 2.40, 1.30, "Regional Motion $E_{k,t}$",
        "$\\|\\Delta X_t\\|_2$ pooled over $M_{k,t}$\nClip Z-Normalization\nOutput: $(B, K, T)$",
        fc="#FFF7ED", ec="#C2410C")

    # Stage 3
    box(6.10, 2.42, 2.60, 1.20, "ROI Feature Pooling",
        "Mask-weighted pooling:\n$F_{k,t}\\;(B, K, C, T)$ +\nGlobal Context $G_t$",
        fc="#F5F3FF", ec="#6D28D9")
    box(6.10, 0.45, 2.60, 1.40, "Stable Region Gate",
        "$\\ell_{k,t}=b_k - \\gamma_k\\sigma(m_{k,t}) + 0.5\\tanh(f_{k,t})$\nGaussian Smooth ($K_s=11$)\n$w_{k,t} = \\mathrm{softmax}_k(\\tilde{\\ell}_{k,t})$",
        fc="#FEF2F2", ec="#B91C1C")

    # Stage 4
    box(9.25, 2.38, 2.40, 1.25, "Temporal BVP Head",
        "$z_t = \\sum_k w_{k,t} F_{k,t} + \\alpha_g G_t$\nDilated 1D-Conv + ELU\nVar-Normed BVP $\\hat{y}$",
        fc="#EFF6FF", ec="#1D4ED8")
    box(9.25, 0.45, 2.40, 1.35, "SKIAv2Loss Objective",
        "$\\mathcal{L}_{\\mathrm{Pearson}} + \\lambda_{\\mathrm{psd}}\\mathcal{L}_{\\mathrm{PSD}}$\n$+\\;\\lambda_{\\mathrm{smooth}}\\mathcal{L}_{\\mathrm{smooth}}$\n$\\mathcal{L}_{\\mathrm{smooth}}=\\mathrm{mean}\\|w_t-w_{t-1}\\|_1$",
        fc="#FFFBEB", ec="#B45309")

    # Arrows
    arrow(2.65, 3.00, 3.10, 3.00)
    arrow(2.65, 2.60, 3.10, 1.35, color="#C2410C")
    arrow(2.65, 1.12, 3.10, 1.12, color="#047857")
    arrow(2.65, 1.45, 6.05, 2.75, color="#047857")
    arrow(5.60, 3.00, 6.05, 3.00)
    arrow(5.60, 1.15, 6.05, 1.15, color="#C2410C")
    arrow(7.40, 2.37, 7.40, 1.90, color="#6D28D9")
    arrow(8.75, 3.00, 9.20, 3.00)
    arrow(8.75, 1.35, 9.20, 2.60, color="#B91C1C")
    arrow(8.75, 1.05, 9.20, 1.05, color="#B45309")
    arrow(10.45, 2.33, 10.45, 1.85, color="#1D4ED8")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_exp_a_chart(out_path: str) -> None:
    df = pd.read_csv(os.path.join(RES_DIR, "exp_a_h1_h2.csv"))
    order = [
        "full_frame_raw",
        "full_frame_static_70",
        "full_frame_bg_removed",
        "face_crop_only",
        "face_static_70",
        "mouth_jaw_suppression",
        "random_equal_area_mask",
        "forehead_cheeks_only",
        "skin_mask_only",
    ]
    labels = [
        "Full Frame\nRaw",
        "Full Frame\nStatic 70%",
        "Full Frame\n+BG Removed",
        "Face Crop\nOnly",
        "Face Crop\n+Static 70%",
        "Mouth/Jaw\nSuppr.",
        "Random\nEqual Area",
        "Forehead+\nCheeks",
        "Skin Mask\nOnly",
    ]
    chrom = df[df["Method"] == "CHROM"].set_index("Variant").loc[order]
    pos = df[df["Method"] == "POS"].set_index("Variant").loc[order]

    x = np.arange(len(order))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.15), dpi=300)

    # Panel A: Signed Pearson correlation (Beer-Lambert polarity aligned across CHROM & POS)
    axes[0].bar(x - w / 2, chrom["Pearson"], w, label="CHROM", color="#2563EB", alpha=0.9)
    axes[0].bar(x + w / 2, pos["Pearson"], w, label="POS", color="#EA580C", alpha=0.9)
    axes[0].axhline(chrom.loc["face_crop_only", "Pearson"], ls="--", lw=1.0, color="#1E40AF", alpha=0.7)
    axes[0].set_title("(a) Signed Waveform Pearson r (N=42, T=800)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Signed Pearson r (higher is better)", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7.2, rotation=24, ha="right")
    axes[0].set_ylim(0, 0.77)
    axes[0].grid(axis="y", ls=":", alpha=0.5)
    axes[0].legend(fontsize=8, loc="upper left")

    for i, (vc, vp) in enumerate(zip(chrom["Pearson"], pos["Pearson"])):
        axes[0].text(i - w / 2 - 0.03, vc + 0.012, f"{vc:.2f}", ha="center", va="bottom", fontsize=5.8, color="#1E3A8A")
        axes[0].text(i + w / 2 + 0.03, vp + 0.012, f"{vp:.2f}", ha="center", va="bottom", fontsize=5.8, color="#9A3412")

    # Panel B: HR MAE
    axes[1].bar(x - w / 2, chrom["HR_MAE"], w, label="CHROM", color="#2563EB", alpha=0.9)
    axes[1].bar(x + w / 2, pos["HR_MAE"], w, label="POS", color="#EA580C", alpha=0.9)
    axes[1].set_title("(b) Heart Rate MAE (bpm, lower is better)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("HR MAE (bpm)", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7.2, rotation=24, ha="right")
    axes[1].set_ylim(0, 38)
    axes[1].grid(axis="y", ls=":", alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper right")

    for i, (vc, vp) in enumerate(zip(chrom["HR_MAE"], pos["HR_MAE"])):
        axes[1].text(i - w / 2 - 0.03, vc + 0.5, f"{vc:.1f}", ha="center", va="bottom", fontsize=5.8, color="#1E3A8A")
        axes[1].text(i + w / 2 + 0.03, vp + 0.5, f"{vp:.1f}", ha="center", va="bottom", fontsize=5.8, color="#9A3412")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_mean_std(s: object) -> tuple:
    parts = [p.strip() for p in str(s).split("±")]
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std


def make_ablation_cross_chart(out_path: str) -> None:
    strata = pd.read_csv(os.path.join(RES_DIR, "exp_motion_strata_multiseed.csv"))
    cross_sum = pd.read_csv(os.path.join(RES_DIR, "exp_cross_dataset_summary.csv"))
    cross_cond = pd.read_csv(os.path.join(RES_DIR, "exp_mmpd_conditions.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.15), dpi=300)

    # Panel A: 3-Seed Motion-Stratified Robustness (Low vs High Mouth Motion, UBFC T=800)
    variants = [
        ("SKIA-v2 Static Prior", "Static\nPrior"),
        ("SKIA-v2 Gauss Off, λ=0", "Gauss Off\nλ=0"),
        ("SKIA-v2 Gauss On, λ=0.05", "Gauss On\nλ=0.05"),
        ("SKIA-v2 Gauss On, λ=0.10 (Full)", "Gauss On\nλ=0.10 (Def.)"),
        ("SKIA-v2 Upper-Face Only (K=3)", "Upper-Face\n(K=3, Val)"),
    ]
    low_m, low_s, high_m, high_s = [], [], [], []
    for v, _ in variants:
        sub = strata[strata["Variant"] == v].set_index("Stratum")
        lm, ls = _parse_mean_std(sub.loc["Low Mouth Motion", "Mean_Pearson"])
        hm, hs = _parse_mean_std(sub.loc["High Mouth Motion", "Mean_Pearson"])
        low_m.append(lm)
        low_s.append(ls)
        high_m.append(hm)
        high_s.append(hs)

    x1 = np.arange(len(variants))
    w1 = 0.36
    axes[0].bar(x1 - w1 / 2, low_m, w1, yerr=low_s, capsize=2.5,
                label="Low Mouth Motion (n=5)", color="#0284C7", alpha=0.9)
    axes[0].bar(x1 + w1 / 2, high_m, w1, yerr=high_s, capsize=2.5,
                label="High Mouth Motion (n=5)", color="#DC2626", alpha=0.9)
    axes[0].set_title("(a) 3-Seed UBFC Mouth-Motion Stratification (T=800)", fontsize=9.8, fontweight="bold")
    axes[0].set_ylabel("Subject Pearson r (mean ± std)", fontsize=9)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels([v[1] for v in variants], fontsize=7.8)
    axes[0].set_ylim(0.75, 0.995)
    axes[0].grid(axis="y", ls=":", alpha=0.5)
    axes[0].legend(fontsize=7.8, loc="upper right")
    for i, (lv, hv) in enumerate(zip(low_m, high_m)):
        drop = hv - lv
        axes[0].text(i - w1 / 2, lv + low_s[i] + 0.003, f"{lv:.3f}", ha="center", va="bottom", fontsize=6.2, color="#0369A1")
        axes[0].text(i + w1 / 2, hv + high_s[i] + 0.003, f"{hv:.3f}", ha="center", va="bottom", fontsize=6.2, color="#991B1B")
        axes[0].text(i + w1 / 2, 0.774, f"$\\Delta={drop:+.3f}$", ha="center", va="center",
                     fontsize=5.4, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.14", fc="#7F1D1D", ec="none", alpha=0.88))

    # Panel B: Balanced Zero-Shot Cross-Dataset HR MAE (showing BOTH resting gains and failure regimes)
    buckets = [
        ("cond", "MCD-rPPG", "Resting (before)", "MCD\nResting"),
        ("cond", "MCD-rPPG", "Post-Exercise (after)", "MCD\nPost-Ex"),
        ("all",  "MCD-rPPG", None, "MCD\nOverall"),
        ("cond", "MMPD", "Talking + LED-high", "MMPD Talk\nLED-high"),
        ("all",  "MMPD", None, "MMPD\nOverall"),
    ]
    models = [
        ("PhysNet", "PhysNet-Lite", "#2563EB"),
        ("TS-CAN", "TS-CAN-Lite", "#F59E0B"),
        ("SKIA-v1 Full", "SKIA-v1 Full", "#64748B"),
        ("SKIA-v2 (Ours)", "SKIA-v2 (Default)", "#059669"),
    ]
    x2 = np.arange(len(buckets))
    w2 = 0.19
    for idx, (m_name, disp_name, col) in enumerate(models):
        vals = []
        for kind, ds_n, cond_n, _ in buckets:
            if kind == "cond":
                row = cross_cond[(cross_cond["Dataset"] == ds_n) &
                                 (cross_cond["Condition"] == cond_n) &
                                 (cross_cond["Model"] == m_name)].iloc[0]
                mae_mean, _ = _parse_mean_std(row["Mean_HR_MAE"])
            else:
                row = cross_sum[(cross_sum["Dataset"] == ds_n) &
                                (cross_sum["Model"] == m_name)].iloc[0]
                mae_mean, _ = _parse_mean_std(row["Subj_HR_MAE"])
            vals.append(mae_mean)
        offset = (idx - 1.5) * w2
        bars = axes[1].bar(x2 + offset, vals, w2, label=disp_name, color=col, alpha=0.92)
        for b, val in zip(bars, vals):
            axes[1].text(b.get_x() + b.get_width() / 2, val + 0.25, f"{val:.1f}",
                         ha="center", va="bottom", fontsize=5.8)

    axes[1].set_title("(b) Zero-Shot Cross-Dataset HR MAE (All & Stratified)", fontsize=9.8, fontweight="bold")
    axes[1].set_ylabel("Mean HR MAE (bpm, lower is better)", fontsize=9)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([b[3] for b in buckets], fontsize=7.8)
    axes[1].set_ylim(0, 19.5)
    axes[1].grid(axis="y", ls=":", alpha=0.5)
    axes[1].legend(fontsize=7.6, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_architecture_diagram(os.path.join(OUT_DIR, "fig_architecture.png"))
    make_exp_a_chart(os.path.join(OUT_DIR, "fig_exp_a_h1_h2.png"))
    make_ablation_cross_chart(os.path.join(OUT_DIR, "fig_ablation_cross.png"))
    print("Generated all 3 paper figures in:", OUT_DIR)


if __name__ == "__main__":
    main()
