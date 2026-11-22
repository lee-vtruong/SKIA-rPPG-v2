"""Unified Rank-B benchmark runner: Experiment A (H1/H2), Experiment B (SOTA), and Ablations."""

import argparse
import gc
import json
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baselines_classical import CLASSICAL_METHODS, extract_rgb_traces
from baselines_deep import build_deep_model
from metrics import (
    bandpass_filter,
    compute_bvp_snr,
    estimate_hr_fft,
    estimate_hr_welch,
    evaluate_predictions,
    paired_significance_tests,
    pearson_corr,
    stitch_subject_predictions,
)
from roi_extractor import (
    EXPERIMENT_A_VARIANTS,
    StabilizedFaceROIExtractor,
    build_variant_masks,
    compute_motion_strata,
)
from skia_dataset import SKIAV2Dataset, load_subject_bvp
from skia_v2_model import SKIANetV2, SKIAv2Loss, neg_pearson_loss
from splits import discover_subjects, get_subject_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_or_extract_subject(
    subj_dir: str,
    extractor: StabilizedFaceROIExtractor,
    size: int = 128,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Return memory-mapped frames + stabilized face crops & anatomical masks for ONE subject."""
    frames_path = os.path.join(subj_dir, "frames.npy")
    crop_path = os.path.join(subj_dir, "face_crops.npy")
    mask_path = os.path.join(subj_dir, "region_masks.npy")
    skin_path = os.path.join(subj_dir, "skin_masks.npy")

    bvp = load_subject_bvp(subj_dir)
    frames = np.load(frames_path, mmap_mode="r") if os.path.exists(frames_path) else None

    if not (os.path.exists(crop_path) and os.path.exists(mask_path) and os.path.exists(skin_path)):
        if frames is None:
            raise FileNotFoundError(f"Missing frames.npy in {subj_dir}")
        ext = extractor.extract_sequence_crops(frames, out_size=size, detect_stride=2)
        np.save(crop_path, ext["crops"].astype(np.float16))
        np.save(mask_path, (ext["region_masks"] > 0.5).astype(np.uint8))
        np.save(skin_path, (ext["skin_masks"] > 0.5).astype(np.uint8))

    crops = np.load(crop_path, mmap_mode="r")
    rmasks = np.load(mask_path, mmap_mode="r")
    smasks = np.load(skin_path, mmap_mode="r")

    T = min(len(crops), len(bvp))
    if frames is not None:
        T = min(T, len(frames))
    if max_frames is not None:
        T = min(T, int(max_frames))

    return {
        "full_frames": frames[:T] if frames is not None else None,
        "face_crops": crops[:T],
        "region_masks": rmasks[:T],
        "skin_masks": smasks[:T],
        "bvp": np.asarray(bvp[:T], dtype=np.float32),
    }


# ==============================================================================
# EXPERIMENT A: H1 (Mouth/Jaw Artifacts) vs H2 (Background / Clothing Removal)
# ==============================================================================

def run_experiment_a(
    data_root: str,
    out_dir: str,
    subjects: Optional[List[str]] = None,
    size: int = 128,
    fs: float = 30.0,
    max_frames: int = 1500,
) -> pd.DataFrame:
    """Evaluate all 8 masking variants with CHROM and POS (streaming one subject at a time)."""
    all_subjects = subjects if subjects is not None else discover_subjects(data_root)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)

    scores: Dict[Tuple[str, str], Dict[str, List[float]]] = {
        (m, v): {"corrs": [], "maes": [], "sqs": [], "snrs": []}
        for m in ("CHROM", "POS")
        for v in EXPERIMENT_A_VARIANTS
    }

    for subj in all_subjects:
        data = _load_or_extract_subject(
            os.path.join(data_root, subj), extractor, size=size, max_frames=max_frames
        )
        gt_bvp = bandpass_filter(data["bvp"], fs=fs)
        hr_t = estimate_hr_welch(gt_bvp, fs=fs)

        for variant in EXPERIMENT_A_VARIANTS:
            if variant.startswith("full_frame") and data["full_frames"] is None:
                continue
            vid, masks = build_variant_masks(
                variant,
                data["full_frames"],
                data["face_crops"],
                data["region_masks"],
                data["skin_masks"],
            )
            r, g, b = extract_rgb_traces(vid, masks=masks)
            for method_name in ("CHROM", "POS"):
                pred_bvp = CLASSICAL_METHODS[method_name](r, g, b, fs)
                c = pearson_corr(pred_bvp, gt_bvp)
                hr_p = estimate_hr_welch(pred_bvp, fs=fs)
                err = abs(hr_p - hr_t)
                snr = compute_bvp_snr(pred_bvp, hr_t, fs=fs)

                bucket = scores[(method_name, variant)]
                bucket["corrs"].append(c)
                bucket["maes"].append(err)
                bucket["sqs"].append(err ** 2)
                bucket["snrs"].append(snr)

    rows = []
    per_variant_subject_scores = scores
    for method_name in ("CHROM", "POS"):
        for variant in EXPERIMENT_A_VARIANTS:
            b = scores[(method_name, variant)]
            if not b["corrs"]:
                continue
            rows.append(
                {
                    "Method": method_name,
                    "Variant": variant,
                    "Pearson": float(np.mean(b["corrs"])),
                    "HR_MAE": float(np.mean(b["maes"])),
                    "HR_RMSE": float(np.sqrt(np.mean(b["sqs"]))),
                    "SNR_dB": float(np.mean(b["snrs"])),
                    "Subjects": len(b["corrs"]),
                }
            )

    df = pd.DataFrame(rows)

    # Paired significance relative to face_crop_only (controls for H2 background removal)
    win_vs_face, p_vs_face, ci_vs_face = [], [], []
    for _, row in df.iterrows():
        m, v = row["Method"], row["Variant"]
        if v == "face_crop_only" or (m, "face_crop_only") not in per_variant_subject_scores:
            win_vs_face.append(np.nan)
            p_vs_face.append(np.nan)
            ci_vs_face.append("")
            continue
        cur = np.array(per_variant_subject_scores[(m, v)]["corrs"])
        base = np.array(per_variant_subject_scores[(m, "face_crop_only")]["corrs"])
        if len(cur) != len(base):
            win_vs_face.append(np.nan)
            p_vs_face.append(np.nan)
            ci_vs_face.append("")
            continue
        sig = paired_significance_tests(cur, base, higher_is_better=True)
        win_vs_face.append(sig["winrate_pct"])
        p_vs_face.append(sig["wilcoxon_p"])
        ci_vs_face.append(f"[{sig['ci95_low']:+.4f}, {sig['ci95_high']:+.4f}]")

    df["Winrate_vs_FaceCrop(%)"] = win_vs_face
    df["Wilcoxon_p_vs_FaceCrop"] = p_vs_face
    df["DeltaPearson_CI95"] = ci_vs_face

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exp_a_h1_h2.csv")
    md_path = os.path.join(out_dir, "exp_a_h1_h2.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment A: H1 (Mouth/Jaw Artifacts) vs H2 (Background/Shirt Removal)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print("\n===== EXPERIMENT A (H1 vs H2) =====")
    print(df.to_string(index=False))
    return df


# ==============================================================================
# SHARED TRAINING / EVALUATION HARNESS (EXPERIMENT B & ABLATIONS)
# ==============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model_speed(
    model: torch.nn.Module,
    device: str,
    clip_len: int = 128,
    size: int = 128,
    is_v2: bool = False,
    iters: int = 30,
) -> Tuple[float, float]:
    """Measure GPU-only inference throughput (FPS) and per-clip latency (ms)."""
    model.eval()
    x = torch.randn(1, 3, clip_len, size, size, device=device)
    m = torch.ones(1, 4, clip_len, size, size, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x, m) if is_v2 else model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x, m) if is_v2 else model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    fps = (iters * clip_len) / max(elapsed, 1e-8)
    latency_ms = (elapsed / iters) * 1000.0
    return fps, latency_ms


def benchmark_end_to_end_fps(
    data_root: str,
    test_subjects: List[str],
    model: torch.nn.Module,
    device: str,
    clip_len: int = 128,
    size: int = 128,
    is_v2: bool = False,
) -> float:
    """Measure end-to-end FPS including stabilized face ROI extraction + tensor prep + inference."""
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
    total_frames = 0
    t0 = time.perf_counter()
    model.eval()

    for subj in test_subjects[:3]:
        frames_path = os.path.join(data_root, subj, "frames.npy")
        if not os.path.exists(frames_path):
            continue
        frames = np.asarray(np.load(frames_path, mmap_mode="r")[:clip_len * 2], dtype=np.float32)
        ext = extractor.extract_sequence_crops(frames, out_size=size, detect_stride=2)
        crops = ext["crops"]
        rmasks = ext["region_masks"]
        for st in range(0, len(crops) - clip_len + 1, clip_len):
            xc = torch.from_numpy(crops[st:st + clip_len]).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            xc = (xc - xc.mean()) / (xc.std() + 1e-6)
            with torch.no_grad():
                if is_v2:
                    mc = torch.from_numpy(rmasks[st:st + clip_len]).permute(1, 0, 2, 3).unsqueeze(0).to(device)
                    _ = model(xc, mc)
                else:
                    _ = model(xc)
            total_frames += clip_len

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return total_frames / max(elapsed, 1e-8)


def _autocast_ctx(device: str):
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)


_GPU_BITS: Dict[str, torch.Tensor] = {}


def _prep_batch_gpu(
    x: torch.Tensor,
    m: torch.Tensor,
    device: str,
    is_v2: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fast GPU uint8 normalization and 4-bit anatomical mask unpacking."""
    if x.dtype == torch.uint8:
        x_d = x.to(device, dtype=torch.float32).mul_(1.0 / 255.0)
        mean = x_d.mean(dim=(1, 2, 3, 4), keepdim=True)
        std = x_d.std(dim=(1, 2, 3, 4), keepdim=True)
        x_d.sub_(mean).div_(std + 1e-6)
    else:
        x_d = x.to(device)
    if not is_v2:
        return x_d, None
    if m.dtype == torch.uint8 and m.ndim == 4:
        key = str(device)
        bits = _GPU_BITS.get(key)
        if bits is None:
            bits = torch.arange(4, device=device, dtype=torch.uint8).view(1, 4, 1, 1, 1)
            _GPU_BITS[key] = bits
        m_d = ((m.to(device).unsqueeze(1) >> bits) & 1).float()
    else:
        m_d = m.to(device).float()
    return x_d, m_d


def _val_clip_pearson(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    is_v2: bool,
) -> float:
    """Fast clip-level Pearson on validation batches for epoch checkpoint selection."""
    model.eval()
    corrs = []
    with torch.no_grad(), _autocast_ctx(device):
        for batch in loader:
            x, m, y = batch[0], batch[1], batch[2]
            x_d, m_d = _prep_batch_gpu(x, m, device, is_v2=is_v2)
            pred = model(x_d, m_d) if is_v2 else model(x_d)
            p_np = pred.float().cpu().numpy()
            y_np = y.numpy()
            for p_i, y_i in zip(p_np, y_np):
                corrs.append(pearson_corr(p_i, y_i))
    return float(np.mean(corrs)) if corrs else 0.0


def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    is_v2: bool,
    fs: float = 30.0,
) -> Dict[str, Any]:
    model.eval()
    preds, targets, subjs, starts = [], [], [], []
    with torch.no_grad(), _autocast_ctx(device):
        for batch in loader:
            x, m, y, subj, st = batch
            x_d, m_d = _prep_batch_gpu(x, m, device, is_v2=is_v2)
            pred = model(x_d, m_d) if is_v2 else model(x_d)
            preds.extend(pred.float().cpu().numpy())
            targets.extend(y.numpy())
            subjs.extend(list(subj))
            starts.extend([int(s) for s in st])
    res = evaluate_predictions(preds, targets, subjects=subjs, starts=starts, fs=fs)
    res["raw_preds"] = preds
    res["raw_targets"] = targets
    res["raw_subjects"] = subjs
    res["raw_starts"] = starts
    return res


def train_and_evaluate_model(
    model_name: str,
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    fs: float,
    ckpt_path: str,
    is_v2: bool = False,
    lambda_psd: float = 0.2,
    lambda_smooth: float = 0.1,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Train a deep rPPG model with strict Val-only checkpointing and test once at the end."""
    model = model_factory().to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        val_r = _val_clip_pearson(model, val_loader, device, is_v2=is_v2)
        test_res = _predict_loader(model, test_loader, device, is_v2=is_v2, fs=fs)
        test_res["best_val_pearson"] = val_r
        test_res["params"] = count_parameters(model)
        return model, test_res

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    v2_loss_fn = SKIAv2Loss(lambda_psd=lambda_psd, lambda_smooth=lambda_smooth, fs=fs)
    tmp_ckpt = ckpt_path + ".tmp"

    best_val_pearson = -999.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x, m, y = batch
            x_d, m_d = _prep_batch_gpu(x, m, device, is_v2=is_v2)
            y_d = y.to(device)
            with _autocast_ctx(device):
                if is_v2:
                    pred, weights = model(x_d, m_d, return_weights=True)
                else:
                    pred = model(x_d)
                    weights = None
            if is_v2:
                loss, _ = v2_loss_fn(pred, y_d, weights=weights)
            else:
                loss = neg_pearson_loss(pred, y_d)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        scheduler.step()
        val_r = _val_clip_pearson(model, val_loader, device, is_v2=is_v2)
        if val_r > best_val_pearson:
            best_val_pearson = val_r
            os.makedirs(os.path.dirname(tmp_ckpt), exist_ok=True)
            torch.save(model.state_dict(), tmp_ckpt)

    if os.path.exists(tmp_ckpt):
        os.replace(tmp_ckpt, ckpt_path)
    else:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_res = _predict_loader(model, test_loader, device, is_v2=is_v2, fs=fs)
    test_res["best_val_pearson"] = best_val_pearson
    test_res["params"] = count_parameters(model)
    return model, test_res


# ==============================================================================
# EXPERIMENT B: Standardized SOTA Comparison across Multiple Seeds
# ==============================================================================

def run_experiment_b(
    data_root: str,
    out_dir: str,
    ckpt_dir: str,
    seeds: List[int],
    epochs: int = 15,
    clip_len: int = 128,
    stride: int = 64,
    size: int = 128,
    batch_size: int = 4,
    lr: float = 3e-4,
    fs: float = 30.0,
) -> pd.DataFrame:
    """Compare Classical & Deep SOTA baselines + SKIA-v1 + SKIA-v2 under strict unified splits."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = False
    subjects = discover_subjects(data_root)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)

    # Pre-cache crops & masks once for all subjects
    for subj in subjects:
        _load_or_extract_subject(os.path.join(data_root, subj), extractor, size=size)

    model_specs = [
        ("PhysNet", lambda: build_deep_model("physnet"), False),
        ("TS-CAN", lambda: build_deep_model("tscan"), False),
        ("EfficientPhys", lambda: build_deep_model("efficientphys"), False),
        ("PhysFormer", lambda: build_deep_model("physformer"), False),
        ("SKIA-v1 Raw", lambda: build_deep_model("skia_raw"), False),
        ("SKIA-v1 Full", lambda: build_deep_model("skia_full"), False),
        ("SKIA-v2 (Ours)", lambda: SKIANetV2(num_regions=4, gate_mode="stable_dynamic"), True),
    ]

    all_records = []

    # Classical baselines on the exact test subjects (deterministic)
    fixed_splits = get_subject_split(subjects, seed=seeds[0], preserve_v1_test=True)
    for m_name in ("CHROM", "POS"):
        for variant in ("face_crop_only", "mouth_jaw_suppression"):
            corrs, maes, sqs, snrs = [], [], [], []
            for subj in fixed_splits["test"]:
                data = _load_or_extract_subject(os.path.join(data_root, subj), extractor, size=size)
                vid, masks = build_variant_masks(
                    variant, data["full_frames"], data["face_crops"], data["region_masks"], data["skin_masks"]
                )
                r, g, b = extract_rgb_traces(vid, masks=masks)
                pred = CLASSICAL_METHODS[m_name](r, g, b, fs)
                gt = bandpass_filter(data["bvp"], fs=fs)
                hr_p = estimate_hr_welch(pred, fs=fs)
                hr_t = estimate_hr_welch(gt, fs=fs)
                err = abs(hr_p - hr_t)
                corrs.append(pearson_corr(pred, gt))
                maes.append(err)
                sqs.append(err ** 2)
                snrs.append(compute_bvp_snr(pred, hr_t, fs=fs))
            all_records.append(
                {
                    "Model": f"{m_name} ({variant})",
                    "Seed": seeds[0],
                    "Clip_Pearson": float(np.mean(corrs)),
                    "Clip_HR_MAE": float(np.mean(maes)),
                    "Subj_Pearson": float(np.mean(corrs)),
                    "Subj_HR_MAE": float(np.mean(maes)),
                    "Subj_HR_RMSE": float(np.sqrt(np.mean(sqs))),
                    "Subj_SNR_dB": float(np.mean(snrs)),
                    "Params": 0,
                    "GPU_FPS": np.nan,
                    "E2E_FPS": np.nan,
                }
            )

    # Deep models across multiple seeds
    for seed in seeds:
        set_seed(seed)
        splits = get_subject_split(subjects, seed=seed, preserve_v1_test=True)
        train_set = SKIAV2Dataset(data_root, splits["train"], clip_len, stride, size, return_masks=True, return_meta=False)
        val_set = SKIAV2Dataset(data_root, splits["val"], clip_len, stride, size, return_masks=True, return_meta=True)
        test_set = SKIAV2Dataset(data_root, splits["test"], clip_len, stride, size, return_masks=True, return_meta=True)
        for ds in (train_set, val_set, test_set):
            ds.fast_gpu_prep = True

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

        for model_label, factory, is_v2 in model_specs:
            set_seed(seed)
            for ds in (train_set, val_set, test_set):
                ds.skip_masks = not is_v2
            safe_name = model_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            ckpt_path = os.path.join(ckpt_dir, f"{safe_name}_seed{seed}.pt")
            model, res = train_and_evaluate_model(
                model_label,
                factory,
                train_loader,
                val_loader,
                test_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                fs=fs,
                ckpt_path=ckpt_path,
                is_v2=is_v2,
            )
            gpu_fps, _ = benchmark_model_speed(model, device, clip_len=clip_len, size=size, is_v2=is_v2)
            e2e_fps = (
                benchmark_end_to_end_fps(
                    data_root, splits["test"], model, device, clip_len=clip_len, size=size, is_v2=is_v2
                )
                if seed == seeds[0]
                else np.nan
            )

            print(
                f"[Seed {seed}] {model_label:16s} | Subj Pearson={res['subject_pearson']:.4f} | "
                f"Subj MAE={res['subject_hr_mae']:.2f} bpm | SNR={res['subject_snr_db']:.2f} dB | "
                f"Params={res['params']:,} | GPU FPS={gpu_fps:.1f}",
                flush=True,
            )
            all_records.append(
                {
                    "Model": model_label,
                    "Seed": seed,
                    "Clip_Pearson": res["clip_pearson"],
                    "Clip_HR_MAE": res["clip_hr_mae"],
                    "Subj_Pearson": res["subject_pearson"],
                    "Subj_HR_MAE": res["subject_hr_mae"],
                    "Subj_HR_RMSE": res["subject_hr_rmse"],
                    "Subj_SNR_dB": res["subject_snr_db"],
                    "Params": res["params"],
                    "GPU_FPS": gpu_fps,
                    "E2E_FPS": e2e_fps,
                }
            )
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(all_records).to_csv(os.path.join(out_dir, "exp_b_sota_per_seed.csv"), index=False)
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        del train_set, val_set, test_set, train_loader, val_loader, test_loader
        gc.collect()

    raw_df = pd.DataFrame(all_records)
    summary_rows = []
    for model_label in raw_df["Model"].unique():
        sub = raw_df[raw_df["Model"] == model_label]
        summary_rows.append(
            {
                "Model": model_label,
                "Subj_Pearson": f"{sub['Subj_Pearson'].mean():.4f} ± {sub['Subj_Pearson'].std(ddof=0):.4f}",
                "Subj_HR_MAE": f"{sub['Subj_HR_MAE'].mean():.2f} ± {sub['Subj_HR_MAE'].std(ddof=0):.2f}",
                "Subj_HR_RMSE": f"{sub['Subj_HR_RMSE'].mean():.2f} ± {sub['Subj_HR_RMSE'].std(ddof=0):.2f}",
                "Subj_SNR_dB": f"{sub['Subj_SNR_dB'].mean():.2f} ± {sub['Subj_SNR_dB'].std(ddof=0):.2f}",
                "Clip_Pearson": f"{sub['Clip_Pearson'].mean():.4f}",
                "Clip_HR_MAE": f"{sub['Clip_HR_MAE'].mean():.2f}",
                "Params": int(sub["Params"].iloc[0]),
                "GPU_FPS": float(sub["GPU_FPS"].mean()) if not sub["GPU_FPS"].isna().all() else np.nan,
                "E2E_FPS": float(sub["E2E_FPS"].dropna().iloc[0]) if not sub["E2E_FPS"].dropna().empty else np.nan,
            }
        )

    sum_df = pd.DataFrame(summary_rows)
    os.makedirs(out_dir, exist_ok=True)
    raw_df.to_csv(os.path.join(out_dir, "exp_b_sota_per_seed.csv"), index=False)
    sum_df.to_csv(os.path.join(out_dir, "exp_b_sota_summary.csv"), index=False)
    with open(os.path.join(out_dir, "exp_b_sota_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Experiment B: Standardized SOTA Comparison (Subject-Independent)\n\n")
        f.write(sum_df.to_markdown(index=False))
        f.write("\n")

    print("\n===== EXPERIMENT B SUMMARY =====")
    print(sum_df.to_string(index=False))
    return sum_df


# ==============================================================================
# ABLATION & MOTION-STRATIFIED ROBUSTNESS ANALYSIS
# ==============================================================================

def run_ablations_and_motion_strata(
    data_root: str,
    out_dir: str,
    ckpt_dir: str,
    seed: int = 42,
    epochs: int = 15,
    clip_len: int = 128,
    stride: int = 64,
    size: int = 128,
    batch_size: int = 4,
    lr: float = 3e-4,
    fs: float = 30.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run SKIA-v2 gating/loss/ROI ablations and Low vs High Mouth-Motion stratification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = False
    subjects = discover_subjects(data_root)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
    for subj in subjects:
        _load_or_extract_subject(os.path.join(data_root, subj), extractor, size=size)

    splits = get_subject_split(subjects, seed=seed, preserve_v1_test=True)
    train_set = SKIAV2Dataset(data_root, splits["train"], clip_len, stride, size, return_masks=True, return_meta=False)
    val_set = SKIAV2Dataset(data_root, splits["val"], clip_len, stride, size, return_masks=True, return_meta=True)
    test_set = SKIAV2Dataset(data_root, splits["test"], clip_len, stride, size, return_masks=True, return_meta=True)
    for ds in (train_set, val_set, test_set):
        ds.fast_gpu_prep = True

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    ablation_configs = [
        ("SKIA-v2 Static Prior", lambda: SKIANetV2(4, gate_mode="static_prior"), 0.2, 0.0),
        ("SKIA-v2 Naive Dynamic (λ=0)", lambda: SKIANetV2(4, gate_mode="naive_dynamic"), 0.2, 0.0),
        ("SKIA-v2 Smooth (λ=0.05)", lambda: SKIANetV2(4, gate_mode="stable_dynamic"), 0.2, 0.05),
        ("SKIA-v2 Smooth (λ=0.10, Full)", lambda: SKIANetV2(4, gate_mode="stable_dynamic"), 0.2, 0.10),
        ("SKIA-v2 No PSD Loss", lambda: SKIANetV2(4, gate_mode="stable_dynamic"), 0.0, 0.10),
        ("SKIA-v2 Upper-Face Only (K=3)", lambda: SKIANetV2(3, gate_mode="stable_dynamic"), 0.2, 0.10),
    ]

    ablation_rows = []
    trained_results: Dict[str, Dict[str, Any]] = {}

    for label, factory, l_psd, l_sm in ablation_configs:
        set_seed(seed)
        safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
        ckpt_path = os.path.join(ckpt_dir, f"abl_{safe}_seed{seed}.pt")
        if label == "SKIA-v2 Smooth (λ=0.10, Full)":
            ours_ckpt = os.path.join(ckpt_dir, f"skia_v2_ours_seed{seed}.pt")
            if os.path.exists(ours_ckpt) and not os.path.exists(ckpt_path):
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                shutil.copyfile(ours_ckpt, ckpt_path)
        model, res = train_and_evaluate_model(
            label,
            factory,
            train_loader,
            val_loader,
            test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            fs=fs,
            ckpt_path=ckpt_path,
            is_v2=True,
            lambda_psd=l_psd,
            lambda_smooth=l_sm,
        )
        trained_results[label] = res
        print(
            f"[Ablation] {label:30s} | Subj Pearson={res['subject_pearson']:.4f} | "
            f"Subj MAE={res['subject_hr_mae']:.2f} bpm | SNR={res['subject_snr_db']:.2f} dB",
            flush=True,
        )
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        ablation_rows.append(
            {
                "Variant": label,
                "Subj_Pearson": res["subject_pearson"],
                "Subj_HR_MAE": res["subject_hr_mae"],
                "Subj_HR_RMSE": res["subject_hr_rmse"],
                "Subj_SNR_dB": res["subject_snr_db"],
                "Clip_Pearson": res["clip_pearson"],
                "Clip_HR_MAE": res["clip_hr_mae"],
            }
        )

    abl_df = pd.DataFrame(ablation_rows)

    # Motion-Stratified Evaluation across test subjects using localized mouth/jaw flow energy
    subj_motion = {}
    for subj in splits["test"]:
        data = _load_or_extract_subject(os.path.join(data_root, subj), extractor, size=size)
        subj_motion[subj] = compute_motion_strata(data["face_crops"], data["region_masks"])["mouth_motion"]

    median_mouth_motion = float(np.median(list(subj_motion.values())))
    strata_rows = []
    for label in ("SKIA-v2 Static Prior", "SKIA-v2 Naive Dynamic (λ=0)", "SKIA-v2 Smooth (λ=0.10, Full)"):
        per_subj = trained_results[label]["per_subject"]
        for stratum_name, cond in [
            ("Low Mouth Motion", lambda m: m <= median_mouth_motion),
            ("High Mouth Motion", lambda m: m > median_mouth_motion),
        ]:
            sub_items = [v for s, v in per_subj.items() if cond(subj_motion[s])]
            if not sub_items:
                continue
            strata_rows.append(
                {
                    "Variant": label,
                    "Stratum": stratum_name,
                    "Subjects": len(sub_items),
                    "Mean_Pearson": float(np.mean([x["pearson"] for x in sub_items])),
                    "Mean_HR_MAE": float(np.mean([x["hr_mae"] for x in sub_items])),
                    "Mean_SNR_dB": float(np.mean([x["snr_db"] for x in sub_items])),
                }
            )

    strata_df = pd.DataFrame(strata_rows)
    os.makedirs(out_dir, exist_ok=True)
    abl_df.to_csv(os.path.join(out_dir, "exp_ablation.csv"), index=False)
    strata_df.to_csv(os.path.join(out_dir, "exp_motion_strata.csv"), index=False)

    print("\n===== ABLATION RESULTS =====")
    print(abl_df.to_string(index=False))
    print("\n===== MOTION-STRATIFIED ROBUSTNESS =====")
    print(strata_df.to_string(index=False))
    return abl_df, strata_df


# ==============================================================================
# CROSS-DATASET ZERO-SHOT & EXPLICIT CONDITION BENCHMARK (MMPD / MCD-rPPG)
# ==============================================================================

def _safe_ckpt_name(model_label: str) -> str:
    return model_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def _max_lag_pearson(pred: np.ndarray, gt: np.ndarray, max_lag: int = 45) -> float:
    """Maximum Pearson correlation across integer shifts in [-max_lag, +max_lag] (±1.5s at 30 Hz)."""
    n = min(len(pred), len(gt))
    if n < 30:
        return pearson_corr(pred, gt)
    best = pearson_corr(pred[:n], gt[:n])
    lim = min(max_lag, n // 3)
    for lag in range(-lim, lim + 1):
        if lag == 0:
            continue
        if lag > 0:
            p_s, g_s = pred[lag:n], gt[: n - lag]
        else:
            k = -lag
            p_s, g_s = pred[: n - k], gt[k:n]
        r = pearson_corr(p_s, g_s)
        if r > best:
            best = r
    return float(best)


def _load_condition_meta(data_root: str, sessions: List[str]) -> Dict[str, Dict[str, Any]]:
    meta_map: Dict[str, Dict[str, Any]] = {}
    for sess in sessions:
        meta_path = os.path.join(data_root, sess, "condition_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_map[sess] = json.load(f)
        else:
            meta_map[sess] = {"session": sess}
    return meta_map


def run_cross_dataset_and_conditions(
    out_dir: str,
    ckpt_dir: str,
    seeds: List[int],
    mmpd_root: str = "data/MMPD/1",
    mcd_root: str = "data/MCD-rPPG/1",
    clip_len: int = 128,
    stride: int = 64,
    size: int = 128,
    batch_size: int = 4,
    fs: float = 30.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate UBFC-trained checkpoints zero-shot on real MMPD and MCD-rPPG conditions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
    datasets = []
    if os.path.isdir(mmpd_root) and discover_subjects(mmpd_root):
        datasets.append(("MMPD", mmpd_root))
    if os.path.isdir(mcd_root) and discover_subjects(mcd_root):
        datasets.append(("MCD-rPPG", mcd_root))

    if not datasets:
        print("No cross-dataset folders found under data/MMPD/1 or data/MCD-rPPG/1; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    model_specs = [
        ("PhysNet", lambda: build_deep_model("physnet"), False),
        ("TS-CAN", lambda: build_deep_model("tscan"), False),
        ("EfficientPhys", lambda: build_deep_model("efficientphys"), False),
        ("PhysFormer", lambda: build_deep_model("physformer"), False),
        ("SKIA-v1 Raw", lambda: build_deep_model("skia_raw"), False),
        ("SKIA-v1 Full", lambda: build_deep_model("skia_full"), False),
        ("SKIA-v2 (Ours)", lambda: SKIANetV2(num_regions=4, gate_mode="stable_dynamic"), True),
    ]

    raw_cross = []
    cond_raw = []

    for ds_name, ds_root in datasets:
        sessions = discover_subjects(ds_root)
        meta_map = _load_condition_meta(ds_root, sessions)

        if ds_name == "MMPD":
            strata_defs = [
                ("Stationary (Low Motion)", lambda m: m.get("motion") == "Stationary"),
                ("Rotation (Head Motion)", lambda m: m.get("motion") == "Rotation"),
                ("Talking (Speech/Mouth Motion)", lambda m: m.get("motion") == "Talking"),
                ("Talking + LED-low", lambda m: m.get("motion") == "Talking" and m.get("light") == "LED-low"),
                ("Talking + LED-high", lambda m: m.get("motion") == "Talking" and m.get("light") == "LED-high"),
            ]
        else:
            strata_defs = [
                ("Resting (before)", lambda m: m.get("step") == "before"),
                ("Post-Exercise (after)", lambda m: m.get("step") == "after"),
            ]

        # Classical zero-shot baselines
        for m_name in ("CHROM", "POS"):
            for variant in ("face_crop_only", "forehead_cheeks_only"):
                per_sess = {}
                for sess in sessions:
                    data = _load_or_extract_subject(os.path.join(ds_root, sess), extractor, size=size)
                    vid, masks = build_variant_masks(
                        variant, data["full_frames"], data["face_crops"], data["region_masks"], data["skin_masks"]
                    )
                    r, g, b = extract_rgb_traces(vid, masks=masks)
                    pred = CLASSICAL_METHODS[m_name](r, g, b, fs)
                    gt = bandpass_filter(data["bvp"], fs=fs)
                    hr_p = estimate_hr_fft(pred, fs=fs)
                    hr_t = estimate_hr_fft(gt, fs=fs)
                    per_sess[sess] = {
                        "pearson": pearson_corr(pred, gt),
                        "lag_pearson": _max_lag_pearson(pred, gt, max_lag=45),
                        "hr_mae": abs(hr_p - hr_t),
                        "snr_db": compute_bvp_snr(pred, hr_t, fs=fs),
                    }
                vals = list(per_sess.values())
                raw_cross.append(
                    {
                        "Dataset": ds_name,
                        "Model": f"{m_name} ({variant})",
                        "Seed": seeds[0],
                        "Sessions": len(vals),
                        "Subj_Pearson": float(np.mean([v["pearson"] for v in vals])),
                        "Subj_Lag_Pearson": float(np.mean([v["lag_pearson"] for v in vals])),
                        "Subj_HR_MAE": float(np.mean([v["hr_mae"] for v in vals])),
                        "Subj_HR_RMSE": float(np.sqrt(np.mean([v["hr_mae"] ** 2 for v in vals]))),
                        "Subj_SNR_dB": float(np.mean([v["snr_db"] for v in vals])),
                        "Clip_Pearson": float(np.mean([v["pearson"] for v in vals])),
                        "Clip_HR_MAE": float(np.mean([v["hr_mae"] for v in vals])),
                    }
                )
                for cond_label, cond_fn in strata_defs:
                    c_vals = [per_sess[s] for s in sessions if s in per_sess and cond_fn(meta_map[s])]
                    if c_vals:
                        cond_raw.append(
                            {
                                "Dataset": ds_name,
                                "Condition": cond_label,
                                "Model": f"{m_name} ({variant})",
                                "Seed": seeds[0],
                                "Sessions": len(c_vals),
                                "Mean_Pearson": float(np.mean([v["pearson"] for v in c_vals])),
                                "Mean_Lag_Pearson": float(np.mean([v["lag_pearson"] for v in c_vals])),
                                "Mean_HR_MAE": float(np.mean([v["hr_mae"] for v in c_vals])),
                                "Mean_SNR_dB": float(np.mean([v["snr_db"] for v in c_vals])),
                            }
                        )

        test_set = SKIAV2Dataset(ds_root, sessions, clip_len, stride, size, return_masks=True, return_meta=True)
        test_set.fast_gpu_prep = True
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        for model_label, factory, is_v2 in model_specs:
            test_set.skip_masks = not is_v2
            for seed in seeds:
                ckpt_path = os.path.join(ckpt_dir, f"{_safe_ckpt_name(model_label)}_seed{seed}.pt")
                if not os.path.exists(ckpt_path):
                    continue
                model = factory().to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                res = _predict_loader(model, test_loader, device, is_v2=is_v2, fs=fs)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

                stitched = stitch_subject_predictions(
                    res["raw_preds"], res["raw_targets"], res["raw_subjects"], res["raw_starts"]
                )
                lag_corrs = []
                for s, (p_full, t_full) in stitched.items():
                    lp = _max_lag_pearson(bandpass_filter(p_full, fs=fs), bandpass_filter(t_full, fs=fs), max_lag=45)
                    res["per_subject"][s]["lag_pearson"] = lp
                    lag_corrs.append(lp)
                subj_lag_r = float(np.mean(lag_corrs)) if lag_corrs else 0.0

                raw_cross.append(
                    {
                        "Dataset": ds_name,
                        "Model": model_label,
                        "Seed": seed,
                        "Sessions": len(res["per_subject"]),
                        "Subj_Pearson": res["subject_pearson"],
                        "Subj_Lag_Pearson": subj_lag_r,
                        "Subj_HR_MAE": res["subject_hr_mae"],
                        "Subj_HR_RMSE": res["subject_hr_rmse"],
                        "Subj_SNR_dB": res["subject_snr_db"],
                        "Clip_Pearson": res["clip_pearson"],
                        "Clip_HR_MAE": res["clip_hr_mae"],
                    }
                )
                print(
                    f"[Cross-{ds_name} Seed {seed}] {model_label:16s} | "
                    f"Pearson={res['subject_pearson']:.4f} | LagPearson={subj_lag_r:.4f} | "
                    f"MAE={res['subject_hr_mae']:.2f} bpm | SNR={res['subject_snr_db']:.2f} dB",
                    flush=True,
                )
                for cond_label, cond_fn in strata_defs:
                    c_vals = [
                        res["per_subject"][s]
                        for s in sessions
                        if s in res["per_subject"] and cond_fn(meta_map[s])
                    ]
                    if c_vals:
                        cond_raw.append(
                            {
                                "Dataset": ds_name,
                                "Condition": cond_label,
                                "Model": model_label,
                                "Seed": seed,
                                "Sessions": len(c_vals),
                                "Mean_Pearson": float(np.mean([v["pearson"] for v in c_vals])),
                                "Mean_Lag_Pearson": float(np.mean([v.get("lag_pearson", 0.0) for v in c_vals])),
                                "Mean_HR_MAE": float(np.mean([v["hr_mae"] for v in c_vals])),
                                "Mean_SNR_dB": float(np.mean([v["snr_db"] for v in c_vals])),
                            }
                        )

        del test_set, test_loader
        gc.collect()

    cross_raw_df = pd.DataFrame(raw_cross)
    cross_sum_rows = []
    for (ds_name, model_label), grp in cross_raw_df.groupby(["Dataset", "Model"], sort=False):
        cross_sum_rows.append(
            {
                "Dataset": ds_name,
                "Model": model_label,
                "Sessions": int(grp["Sessions"].iloc[0]),
                "Subj_Pearson": f"{grp['Subj_Pearson'].mean():.4f} ± {grp['Subj_Pearson'].std(ddof=0):.4f}",
                "Subj_Lag_Pearson": f"{grp['Subj_Lag_Pearson'].mean():.4f} ± {grp['Subj_Lag_Pearson'].std(ddof=0):.4f}",
                "Subj_HR_MAE": f"{grp['Subj_HR_MAE'].mean():.2f} ± {grp['Subj_HR_MAE'].std(ddof=0):.2f}",
                "Subj_HR_RMSE": f"{grp['Subj_HR_RMSE'].mean():.2f} ± {grp['Subj_HR_RMSE'].std(ddof=0):.2f}",
                "Subj_SNR_dB": f"{grp['Subj_SNR_dB'].mean():.2f} ± {grp['Subj_SNR_dB'].std(ddof=0):.2f}",
                "Clip_Pearson": f"{grp['Clip_Pearson'].mean():.4f}",
                "Clip_HR_MAE": f"{grp['Clip_HR_MAE'].mean():.2f}",
            }
        )
    cross_sum_df = pd.DataFrame(cross_sum_rows)

    cond_raw_df = pd.DataFrame(cond_raw)
    cond_sum_rows = []
    if not cond_raw_df.empty:
        for (ds_name, cond_label, model_label), grp in cond_raw_df.groupby(
            ["Dataset", "Condition", "Model"], sort=False
        ):
            cond_sum_rows.append(
                {
                    "Dataset": ds_name,
                    "Condition": cond_label,
                    "Model": model_label,
                    "Sessions": int(grp["Sessions"].iloc[0]),
                    "Mean_Pearson": f"{grp['Mean_Pearson'].mean():.4f} ± {grp['Mean_Pearson'].std(ddof=0):.4f}",
                    "Mean_Lag_Pearson": f"{grp['Mean_Lag_Pearson'].mean():.4f} ± {grp['Mean_Lag_Pearson'].std(ddof=0):.4f}",
                    "Mean_HR_MAE": f"{grp['Mean_HR_MAE'].mean():.2f} ± {grp['Mean_HR_MAE'].std(ddof=0):.2f}",
                    "Mean_SNR_dB": f"{grp['Mean_SNR_dB'].mean():.2f} ± {grp['Mean_SNR_dB'].std(ddof=0):.2f}",
                }
            )
    cond_sum_df = pd.DataFrame(cond_sum_rows)

    os.makedirs(out_dir, exist_ok=True)
    cross_raw_df.to_csv(os.path.join(out_dir, "exp_cross_dataset_per_seed.csv"), index=False)
    cross_sum_df.to_csv(os.path.join(out_dir, "exp_cross_dataset_summary.csv"), index=False)
    cond_sum_df.to_csv(os.path.join(out_dir, "exp_mmpd_conditions.csv"), index=False)
    with open(os.path.join(out_dir, "exp_cross_dataset_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Cross-Dataset Zero-Shot Generalization (Trained on UBFC-rPPG)\n\n")
        f.write(cross_sum_df.to_markdown(index=False))
        f.write("\n")
    with open(os.path.join(out_dir, "exp_mmpd_conditions.md"), "w", encoding="utf-8") as f:
        f.write("# Condition-Stratified Zero-Shot Robustness (MMPD & MCD-rPPG)\n\n")
        f.write(cond_sum_df.to_markdown(index=False))
        f.write("\n")

    print("\n===== CROSS-DATASET ZERO-SHOT SUMMARY =====")
    print(cross_sum_df.to_string(index=False))
    print("\n===== CONDITION-STRATIFIED ROBUSTNESS =====")
    print(cond_sum_df.to_string(index=False))
    return cross_sum_df, cond_sum_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--exp", type=str, default="all", choices=["A", "B", "ablation", "cross", "all"])
    parser.add_argument("--out-dir", type=str, default="../results/v2")
    parser.add_argument("--ckpt-dir", type=str, default="../checkpoints/v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--clip-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fs", type=float, default=30.0)
    args = parser.parse_args()

    if args.exp in ("A", "all"):
        run_experiment_a(args.data_root, args.out_dir, size=args.size, fs=args.fs)
    if args.exp in ("B", "all"):
        run_experiment_b(
            args.data_root,
            args.out_dir,
            args.ckpt_dir,
            seeds=args.seeds,
            epochs=args.epochs,
            clip_len=args.clip_len,
            stride=args.stride,
            size=args.size,
            batch_size=args.batch_size,
            lr=args.lr,
            fs=args.fs,
        )
    if args.exp in ("ablation", "all"):
        run_ablations_and_motion_strata(
            args.data_root,
            args.out_dir,
            args.ckpt_dir,
            seed=args.seeds[0],
            epochs=args.epochs,
            clip_len=args.clip_len,
            stride=args.stride,
            size=args.size,
            batch_size=args.batch_size,
            lr=args.lr,
            fs=args.fs,
        )
    if args.exp in ("cross", "all"):
        run_cross_dataset_and_conditions(
            out_dir=args.out_dir,
            ckpt_dir=args.ckpt_dir,
            seeds=args.seeds,
            clip_len=args.clip_len,
            stride=args.stride,
            size=args.size,
            batch_size=args.batch_size,
            fs=args.fs,
        )


if __name__ == "__main__":
    main()
