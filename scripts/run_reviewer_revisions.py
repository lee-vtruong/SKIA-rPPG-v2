"""Run all reviewer-requested experiments and audits on real T=800 UBFC, MCD-rPPG, and MMPD data.

Parts:
  A. Fixed-polarity POS across Experiment A (both Discovery N=32 and Full N=42 with
     9th variant `full_frame_bg_removed`), Experiment B, and Cross-Dataset benchmarks.
  B. Multi-seed Spectral vs. Oximeter Sensor HR and Multi-Window (4.27s/10s/20s/26.7s)
     plus Validation-Pearson audit across all Experiment B models and classical baselines.
  C. 3-seed (42, 43, 44) complete ablation and 2x3 factorial decomposition
     (Gaussian logit filter {Off, On} x L_smooth {0, 0.05, 0.10}) + Static Prior +
     Upper-Face (K=3), with Validation-Pearson model selection, 3-seed mouth-motion
     stratification, and 3-seed cross-dataset evaluation on MCD-rPPG and MMPD.
"""

import argparse
import gc
import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from baselines_classical import CLASSICAL_METHODS, extract_rgb_traces
from baselines_deep import build_deep_model
from metrics import (
    bandpass_filter,
    compute_bvp_snr,
    estimate_hr_fft,
    estimate_hr_welch,
    pearson_corr,
    stitch_subject_predictions,
)
from roi_extractor import (
    StabilizedFaceROIExtractor,
    build_variant_masks,
    compute_motion_strata,
)
from run_experiments import (
    _load_condition_meta,
    _load_or_extract_subject,
    _max_lag_pearson,
    _predict_loader,
    _safe_ckpt_name,
    _val_clip_pearson,
    run_cross_dataset_and_conditions,
    run_experiment_a,
    run_experiment_b,
    set_seed,
    train_and_evaluate_model,
)
from skia_dataset import SKIAV2Dataset, load_subject_bvp
from skia_v2_model import SKIANetV2
from splits import UBFC_V1_TEST_SUBJECTS, discover_subjects, get_subject_split


def _make_split_loaders(
    data_root: str,
    splits: Dict[str, List[str]],
    clip_len: int = 128,
    stride: int = 64,
    size: int = 128,
    batch_size: int = 4,
    skip_masks: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set = SKIAV2Dataset(data_root, splits["train"], clip_len, stride, size, return_masks=True, return_meta=False)
    val_set = SKIAV2Dataset(data_root, splits["val"], clip_len, stride, size, return_masks=True, return_meta=False)
    test_set = SKIAV2Dataset(data_root, splits["test"], clip_len, stride, size, return_masks=True, return_meta=True)
    for ds in (train_set, val_set, test_set):
        ds.fast_gpu_prep = True
        ds.skip_masks = skip_masks
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def _load_sensor_hr(subj_dir: str, max_len: int) -> Optional[float]:
    gt_txt = os.path.join(subj_dir, "ground_truth.txt")
    if not os.path.exists(gt_txt):
        return None
    gt = np.loadtxt(gt_txt)
    if gt.ndim == 2 and gt.shape[0] >= 2:
        n = min(max_len, gt.shape[1])
        return float(np.mean(gt[1, :n]))
    return None


def _window_spectral_mae(p_filt: np.ndarray, t_filt: np.ndarray, win_len: int, stride: int, fs: float) -> float:
    n = min(len(p_filt), len(t_filt))
    if win_len >= n:
        return abs(estimate_hr_fft(p_filt[:n], fs=fs) - estimate_hr_fft(t_filt[:n], fs=fs))
    errs = []
    for st in range(0, n - win_len + 1, stride):
        hp = estimate_hr_fft(p_filt[st:st + win_len], fs=fs)
        ht = estimate_hr_fft(t_filt[st:st + win_len], fs=fs)
        errs.append(abs(hp - ht))
    return float(np.mean(errs)) if errs else abs(estimate_hr_fft(p_filt, fs=fs) - estimate_hr_fft(t_filt, fs=fs))


def run_part_a(data_root: str, out_dir: str, ckpt_dir: str, seeds: List[int], mmpd_root: str, mcd_root: str) -> None:
    print("\n================ PART A: EXP A (DISCOVERY-32 & FULL-42) + FIXED POS ================", flush=True)
    all_subjects = discover_subjects(data_root)
    disc_subjects = [s for s in all_subjects if s not in set(UBFC_V1_TEST_SUBJECTS)]

    tmp_disc_dir = os.path.join(out_dir, "_tmp_disc32")
    os.makedirs(tmp_disc_dir, exist_ok=True)
    run_experiment_a(data_root, tmp_disc_dir, subjects=disc_subjects, size=128, fs=30.0)
    shutil.copyfile(os.path.join(tmp_disc_dir, "exp_a_h1_h2.csv"), os.path.join(out_dir, "exp_a_discovery32.csv"))
    with open(os.path.join(tmp_disc_dir, "exp_a_h1_h2.md"), "r", encoding="utf-8") as f:
        md_txt = f.read()
    with open(os.path.join(out_dir, "exp_a_discovery32.md"), "w", encoding="utf-8") as f:
        f.write(md_txt.replace(
            "# Experiment A: H1 (Mouth/Jaw Artifacts) vs H2 (Background/Shirt Removal)",
            f"# Experiment A (Discovery Split Only, N={len(disc_subjects)}, Excluding Held-Out Test Split)",
        ))
    shutil.rmtree(tmp_disc_dir, ignore_errors=True)

    run_experiment_a(data_root, out_dir, subjects=all_subjects, size=128, fs=30.0)
    run_experiment_b(
        data_root, out_dir, ckpt_dir, seeds=seeds, epochs=20,
        clip_len=128, stride=64, size=128, batch_size=4, lr=3e-4, fs=30.0,
    )
    run_cross_dataset_and_conditions(
        out_dir, ckpt_dir, seeds=seeds, mmpd_root=mmpd_root, mcd_root=mcd_root,
        clip_len=128, stride=64, size=128, batch_size=4, fs=30.0,
    )


def run_part_b(data_root: str, out_dir: str, ckpt_dir: str, seeds: List[int], fs: float = 30.0) -> None:
    print("\n================ PART B: SENSOR HR, MULTI-WINDOW & VAL_R AUDIT ================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = discover_subjects(data_root)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
    fixed_splits = get_subject_split(subjects, seed=seeds[0], preserve_v1_test=True)

    # Pre-cache sensor HRs on the 10 test subjects
    sensor_hr_map: Dict[str, float] = {}
    gt_bvp_map: Dict[str, np.ndarray] = {}
    for subj in fixed_splits["test"]:
        s_dir = os.path.join(data_root, subj)
        data = _load_or_extract_subject(s_dir, extractor, size=128)
        T_s = len(data["face_crops"])
        gt_bvp_map[subj] = data["bvp"][:T_s]
        s_hr = _load_sensor_hr(s_dir, T_s)
        if s_hr is not None:
            sensor_hr_map[subj] = s_hr

    gt_spec_vs_sensor = []
    for subj, bvp in gt_bvp_map.items():
        if subj in sensor_hr_map:
            gt_fft_hr = estimate_hr_fft(bandpass_filter(bvp, fs=fs), fs=fs)
            gt_spec_vs_sensor.append(abs(gt_fft_hr - sensor_hr_map[subj]))
    if gt_spec_vs_sensor:
        print(
            f"[Audit] Reference GT BVP FFT HR vs Oximeter Sensor HR (N={len(gt_spec_vs_sensor)}): "
            f"MAE = {np.mean(gt_spec_vs_sensor):.2f} bpm",
            flush=True,
        )

    rows = []
    # Classical deterministic baselines on the 10 test subjects
    for m_name in ("CHROM", "POS"):
        m_fn = CLASSICAL_METHODS[m_name]
        for variant in ("face_crop_only", "mouth_jaw_suppression"):
            corrs, snrs, m_26, m_w, m_20, m_10, m_4, m_sens = [], [], [], [], [], [], [], []
            near_zero = 0
            for subj in fixed_splits["test"]:
                s_dir = os.path.join(data_root, subj)
                data = _load_or_extract_subject(s_dir, extractor, size=128)
                vid, mask = build_variant_masks(
                    variant, data["full_frames"], data["face_crops"], data["region_masks"], data["skin_masks"]
                )
                r, g, b = extract_rgb_traces(vid, masks=mask)
                n_s = min(len(r), len(data["bvp"]))
                p_f = bandpass_filter(m_fn(r[:n_s], g[:n_s], b[:n_s], fs), fs=fs)
                t_f = bandpass_filter(data["bvp"][:n_s], fs=fs)
                corrs.append(pearson_corr(p_f, t_f))
                hp = estimate_hr_fft(p_f, fs=fs)
                ht = estimate_hr_fft(t_f, fs=fs)
                err_26 = abs(hp - ht)
                if err_26 < 0.25:
                    near_zero += 1
                m_26.append(err_26)
                m_w.append(abs(estimate_hr_welch(p_f, fs=fs) - estimate_hr_welch(t_f, fs=fs)))
                m_20.append(_window_spectral_mae(p_f, t_f, 600, 200, fs))
                m_10.append(_window_spectral_mae(p_f, t_f, 300, 150, fs))
                m_4.append(_window_spectral_mae(p_f, t_f, 128, 64, fs))
                snrs.append(compute_bvp_snr(p_f, ht, fs=fs))
                if subj in sensor_hr_map:
                    m_sens.append(abs(hp - sensor_hr_map[subj]))
            rows.append({
                "Model": f"{m_name} ({variant})",
                "Seed": 42,
                "Val_Pearson": np.nan,
                "Subj_Pearson": float(np.mean(corrs)),
                "Subj_SNR_dB": float(np.mean(snrs)),
                "MAE_Spec_26.7s": float(np.mean(m_26)),
                "MAE_Welch_26.7s": float(np.mean(m_w)),
                "MAE_Spec_20s": float(np.mean(m_20)),
                "MAE_Spec_10s": float(np.mean(m_10)),
                "MAE_Spec_4.27s": float(np.mean(m_4)),
                "MAE_vs_Sensor": float(np.mean(m_sens)),
                "NearZero_lt0.25bpm": int(near_zero),
            })

    model_specs = [
        ("PhysNet", lambda: build_deep_model("physnet"), False),
        ("TS-CAN", lambda: build_deep_model("tscan"), False),
        ("EfficientPhys", lambda: build_deep_model("efficientphys"), False),
        ("PhysFormer", lambda: build_deep_model("physformer"), False),
        ("SKIA-v1 Raw", lambda: build_deep_model("skia_raw"), False),
        ("SKIA-v1 Full", lambda: build_deep_model("skia_full"), False),
        ("SKIA-v2 (Ours)", lambda: SKIANetV2(num_regions=4, gate_mode="stable_dynamic"), True),
    ]

    for seed in seeds:
        splits = get_subject_split(subjects, seed=seed, preserve_v1_test=True)
        _, val_loader, test_loader = _make_split_loaders(
            data_root, splits, clip_len=128, stride=64, size=128, batch_size=8, skip_masks=False
        )
        for label, factory, is_v2 in model_specs:
            val_loader.dataset.skip_masks = not is_v2
            test_loader.dataset.skip_masks = not is_v2
            ckpt_path = os.path.join(ckpt_dir, f"{_safe_ckpt_name(label)}_seed{seed}.pt")
            model = factory().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            val_r = _val_clip_pearson(model, val_loader, device, is_v2=is_v2)
            res = _predict_loader(model, test_loader, device, is_v2=is_v2, fs=fs)
            stitched = stitch_subject_predictions(
                res["raw_preds"], res["raw_targets"], res["raw_subjects"], res["raw_starts"]
            )
            m_26, m_w, m_20, m_10, m_4, m_sens = [], [], [], [], [], []
            near_zero = 0
            for subj, (p_full, t_full) in sorted(stitched.items()):
                p_f = bandpass_filter(p_full, fs=fs)
                t_f = bandpass_filter(t_full, fs=fs)
                hp = estimate_hr_fft(p_f, fs=fs)
                ht = estimate_hr_fft(t_f, fs=fs)
                err_26 = abs(hp - ht)
                if err_26 < 0.25:
                    near_zero += 1
                m_26.append(err_26)
                m_w.append(abs(estimate_hr_welch(p_f, fs=fs) - estimate_hr_welch(t_f, fs=fs)))
                m_20.append(_window_spectral_mae(p_f, t_f, 600, 200, fs))
                m_10.append(_window_spectral_mae(p_f, t_f, 300, 150, fs))
                m_4.append(_window_spectral_mae(p_f, t_f, 128, 64, fs))
                if subj in sensor_hr_map:
                    m_sens.append(abs(hp - sensor_hr_map[subj]))
            rows.append({
                "Model": label,
                "Seed": seed,
                "Val_Pearson": val_r,
                "Subj_Pearson": res["subject_pearson"],
                "Subj_SNR_dB": res["subject_snr_db"],
                "MAE_Spec_26.7s": float(np.mean(m_26)),
                "MAE_Welch_26.7s": float(np.mean(m_w)),
                "MAE_Spec_20s": float(np.mean(m_20)),
                "MAE_Spec_10s": float(np.mean(m_10)),
                "MAE_Spec_4.27s": float(np.mean(m_4)),
                "MAE_vs_Sensor": float(np.mean(m_sens)),
                "NearZero_lt0.25bpm": int(near_zero),
            })
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(os.path.join(out_dir, "exp_hr_window_sensor_audit_per_seed.csv"), index=False)

    sum_rows = []
    for label, grp in df_raw.groupby("Model", sort=False):
        fmt = lambda col, dec=2: (
            f"{grp[col].mean():.{dec}f}" if len(grp) == 1
            else f"{grp[col].mean():.{dec}f} ± {grp[col].std(ddof=0):.{dec}f}"
        )
        val_str = "—" if grp["Val_Pearson"].isna().all() else fmt("Val_Pearson", 4)
        sum_rows.append({
            "Model": label,
            "Val_Pearson": val_str,
            "Subj_Pearson": fmt("Subj_Pearson", 4),
            "MAE_Spec_26.7s": fmt("MAE_Spec_26.7s"),
            "MAE_Welch_26.7s": fmt("MAE_Welch_26.7s"),
            "MAE_Spec_20s": fmt("MAE_Spec_20s"),
            "MAE_Spec_10s": fmt("MAE_Spec_10s"),
            "MAE_Spec_4.27s": fmt("MAE_Spec_4.27s"),
            "MAE_vs_Sensor": fmt("MAE_vs_Sensor"),
            "NearZero_Avg(/10)": f"{grp['NearZero_lt0.25bpm'].mean():.1f}",
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_dir, "exp_hr_window_sensor_audit.csv"), index=False)
    with open(os.path.join(out_dir, "exp_hr_window_sensor_audit.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-Window Spectral vs. Pulse Oximeter Sensor HR Audit (UBFC Test Split, N=10)\n\n")
        if gt_spec_vs_sensor:
            f.write(
                f"Reference GT BVP FFT HR vs. Row-1 Oximeter Sensor HR MAE = "
                f"{np.mean(gt_spec_vs_sensor):.2f} bpm.\n\n"
            )
        f.write(df_sum.to_markdown(index=False))
        f.write("\n")
    print(df_sum.to_string(index=False), flush=True)


def run_part_c(
    data_root: str,
    out_dir: str,
    ckpt_dir: str,
    seeds: List[int],
    mmpd_root: str,
    mcd_root: str,
    epochs: int = 20,
    fs: float = 30.0,
) -> None:
    print("\n================ PART C: 3-SEED ABLATION, 2x3 FACTORIAL, VAL SELECTION & CROSS-DS ================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = discover_subjects(data_root)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)

    # Pre-cache mouth motion strata & sensor HRs on the 10 held-out UBFC test subjects
    subj_motion: Dict[str, float] = {}
    sensor_hr_map: Dict[str, float] = {}
    for subj in UBFC_V1_TEST_SUBJECTS:
        s_dir = os.path.join(data_root, subj)
        data = _load_or_extract_subject(s_dir, extractor, size=128)
        subj_motion[subj] = compute_motion_strata(data["face_crops"], data["region_masks"])["mouth_motion"]
        s_hr = _load_sensor_hr(s_dir, len(data["face_crops"]))
        if s_hr is not None:
            sensor_hr_map[subj] = s_hr
    median_mouth_motion = float(np.median(list(subj_motion.values())))

    # Preload cross-dataset DataLoaders once to avoid redundant disk I/O
    cross_loaders: List[Tuple[str, DataLoader, List[str], Dict[str, Any], List[Tuple[str, Callable]]]] = []
    for ds_name, ds_root in (("MCD-rPPG", mcd_root), ("MMPD", mmpd_root)):
        if not (os.path.isdir(ds_root) and discover_subjects(ds_root)):
            continue
        sessions = discover_subjects(ds_root)
        meta_map = _load_condition_meta(ds_root, sessions)
        if ds_name == "MCD-rPPG":
            strata_defs = [
                ("Resting (before)", lambda m: m.get("step") == "before"),
                ("Post-Exercise (after)", lambda m: m.get("step") == "after"),
            ]
        else:
            strata_defs = [
                ("Stationary (Low Motion)", lambda m: m.get("motion") == "Stationary"),
                ("Rotation (Head Motion)", lambda m: m.get("motion") == "Rotation"),
                ("Talking + LED-low", lambda m: m.get("motion") == "Talking" and m.get("light") == "LED-low"),
                ("Talking + LED-high", lambda m: m.get("motion") == "Talking" and m.get("light") == "LED-high"),
            ]
        ds = SKIAV2Dataset(ds_root, sessions, 128, 64, 128, return_masks=True, return_meta=True)
        ds.fast_gpu_prep = True
        ds.skip_masks = False
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
        cross_loaders.append((ds_name, loader, sessions, meta_map, strata_defs))

    # Core 8 variants covering Static Prior, full 2x3 (Gauss {Off, On} x L_smooth {0, 0.05, 0.10}), and Upper-Face K=3
    ablation_specs = [
        {
            "label": "SKIA-v2 Static Prior",
            "gauss": "—",
            "l_sm": 0.0,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="static_prior"),
            "ckpt_fn": lambda s: (
                os.path.join(ckpt_dir, "abl_skia-v2_static_prior_seed42.pt")
                if s == 42 and os.path.exists(os.path.join(ckpt_dir, "abl_skia-v2_static_prior_seed42.pt"))
                else os.path.join(ckpt_dir, f"abl_static_prior_seed{s}.pt")
            ),
        },
        {
            "label": "SKIA-v2 Gauss Off, λ=0",
            "gauss": "Off",
            "l_sm": 0.0,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="naive_dynamic"),
            "ckpt_fn": lambda s: (
                os.path.join(ckpt_dir, "abl_skia-v2_naive_dynamic_λ0_seed42.pt")
                if s == 42 and os.path.exists(os.path.join(ckpt_dir, "abl_skia-v2_naive_dynamic_λ0_seed42.pt"))
                else os.path.join(ckpt_dir, f"abl_gauss_off_l0.00_seed{s}.pt")
            ),
        },
        {
            "label": "SKIA-v2 Gauss Off, λ=0.05",
            "gauss": "Off",
            "l_sm": 0.05,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="naive_dynamic"),
            "ckpt_fn": lambda s: os.path.join(ckpt_dir, f"abl_gauss_off_l0.05_seed{s}.pt"),
        },
        {
            "label": "SKIA-v2 Gauss Off, λ=0.10",
            "gauss": "Off",
            "l_sm": 0.10,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="naive_dynamic"),
            "ckpt_fn": lambda s: os.path.join(ckpt_dir, f"abl_gauss_off_l0.10_seed{s}.pt"),
        },
        {
            "label": "SKIA-v2 Gauss On, λ=0",
            "gauss": "On",
            "l_sm": 0.0,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="stable_dynamic"),
            "ckpt_fn": lambda s: os.path.join(ckpt_dir, f"abl_gauss_on_l0.00_seed{s}.pt"),
        },
        {
            "label": "SKIA-v2 Gauss On, λ=0.05",
            "gauss": "On",
            "l_sm": 0.05,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="stable_dynamic"),
            "ckpt_fn": lambda s: (
                os.path.join(ckpt_dir, "abl_skia-v2_smooth_λ0.05_seed42.pt")
                if s == 42 and os.path.exists(os.path.join(ckpt_dir, "abl_skia-v2_smooth_λ0.05_seed42.pt"))
                else os.path.join(ckpt_dir, f"abl_gauss_on_l0.05_seed{s}.pt")
            ),
        },
        {
            "label": "SKIA-v2 Gauss On, λ=0.10 (Full)",
            "gauss": "On",
            "l_sm": 0.10,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=4, gate_mode="stable_dynamic"),
            "ckpt_fn": lambda s: os.path.join(ckpt_dir, f"skia_v2_ours_seed{s}.pt"),
        },
        {
            "label": "SKIA-v2 Upper-Face Only (K=3)",
            "gauss": "On",
            "l_sm": 0.10,
            "l_psd": 0.2,
            "factory": lambda: SKIANetV2(num_regions=3, gate_mode="stable_dynamic"),
            "ckpt_fn": lambda s: (
                os.path.join(ckpt_dir, "abl_skia-v2_upper-face_only_k3_seed42.pt")
                if s == 42 and os.path.exists(os.path.join(ckpt_dir, "abl_skia-v2_upper-face_only_k3_seed42.pt"))
                else os.path.join(ckpt_dir, f"abl_upper_face_k3_seed{s}.pt")
            ),
        },
    ]

    abl_rows, strata_rows, cross_rows, cond_rows = [], [], [], []

    for seed in seeds:
        splits = get_subject_split(subjects, seed=seed, preserve_v1_test=True)
        train_loader, val_loader, test_loader = _make_split_loaders(
            data_root, splits, clip_len=128, stride=64, size=128, batch_size=4, skip_masks=False
        )
        for spec in ablation_specs:
            t0 = time.time()
            set_seed(seed)
            ckpt_path = spec["ckpt_fn"](seed)
            existed = os.path.exists(ckpt_path)
            model, res = train_and_evaluate_model(
                spec["label"],
                spec["factory"],
                train_loader,
                val_loader,
                test_loader,
                device=device,
                epochs=epochs,
                lr=3e-4,
                fs=fs,
                ckpt_path=ckpt_path,
                is_v2=True,
                lambda_psd=spec["l_psd"],
                lambda_smooth=spec["l_sm"],
            )
            sens_errs = [
                abs(v["hr_pred"] - sensor_hr_map[s])
                for s, v in res["per_subject"].items()
                if s in sensor_hr_map
            ]
            abl_rows.append({
                "Variant": spec["label"],
                "Gauss_Filter": spec["gauss"],
                "Lambda_Smooth": spec["l_sm"],
                "Seed": seed,
                "Val_Pearson": res["best_val_pearson"],
                "Subj_Pearson": res["subject_pearson"],
                "Subj_HR_MAE": res["subject_hr_mae"],
                "Subj_HR_RMSE": res["subject_hr_rmse"],
                "Subj_SNR_dB": res["subject_snr_db"],
                "Clip_Pearson": res["clip_pearson"],
                "Sensor_HR_MAE": float(np.mean(sens_errs)) if sens_errs else np.nan,
            })

            # Motion strata on UBFC Test
            per_subj = res["per_subject"]
            for stratum_name, cond in (
                ("Low Mouth Motion", lambda m: m <= median_mouth_motion),
                ("High Mouth Motion", lambda m: m > median_mouth_motion),
            ):
                sub_items = [v for s, v in per_subj.items() if cond(subj_motion[s])]
                if sub_items:
                    strata_rows.append({
                        "Variant": spec["label"],
                        "Seed": seed,
                        "Stratum": stratum_name,
                        "Subjects": len(sub_items),
                        "Mean_Pearson": float(np.mean([x["pearson"] for x in sub_items])),
                        "Mean_HR_MAE": float(np.mean([x["hr_mae"] for x in sub_items])),
                        "Mean_SNR_dB": float(np.mean([x["snr_db"] for x in sub_items])),
                    })

            # Cross-dataset evaluation for this exact checkpoint
            for ds_name, c_loader, sessions, meta_map, strata_defs in cross_loaders:
                c_res = _predict_loader(model, c_loader, device, is_v2=True, fs=fs)
                stitched = stitch_subject_predictions(
                    c_res["raw_preds"], c_res["raw_targets"], c_res["raw_subjects"], c_res["raw_starts"]
                )
                lag_corrs = []
                for s, (p_full, t_full) in stitched.items():
                    lp = _max_lag_pearson(bandpass_filter(p_full, fs=fs), bandpass_filter(t_full, fs=fs), max_lag=45)
                    c_res["per_subject"][s]["lag_pearson"] = lp
                    lag_corrs.append(lp)
                cross_rows.append({
                    "Dataset": ds_name,
                    "Variant": spec["label"],
                    "Seed": seed,
                    "Sessions": len(c_res["per_subject"]),
                    "Subj_Pearson": c_res["subject_pearson"],
                    "Oracle_Lag_Pearson": float(np.mean(lag_corrs)) if lag_corrs else 0.0,
                    "Subj_HR_MAE": c_res["subject_hr_mae"],
                    "Subj_HR_RMSE": c_res["subject_hr_rmse"],
                    "Subj_SNR_dB": c_res["subject_snr_db"],
                })
                for cond_label, cond_fn in strata_defs:
                    c_vals = [
                        c_res["per_subject"][s]
                        for s in sessions
                        if s in c_res["per_subject"] and cond_fn(meta_map[s])
                    ]
                    if c_vals:
                        cond_rows.append({
                            "Dataset": ds_name,
                            "Condition": cond_label,
                            "Variant": spec["label"],
                            "Seed": seed,
                            "Sessions": len(c_vals),
                            "Mean_Pearson": float(np.mean([v["pearson"] for v in c_vals])),
                            "Oracle_Lag_Pearson": float(np.mean([v.get("lag_pearson", 0.0) for v in c_vals])),
                            "Mean_HR_MAE": float(np.mean([v["hr_mae"] for v in c_vals])),
                            "Mean_SNR_dB": float(np.mean([v["snr_db"] for v in c_vals])),
                        })

            elapsed = time.time() - t0
            tag = "cached" if existed else "trained"
            print(
                f"[Abl Seed {seed} | {tag:7s} {elapsed:5.1f}s] {spec['label']:32s} | "
                f"Val_r={res['best_val_pearson']:.4f} | Test_r={res['subject_pearson']:.4f} | "
                f"MAE={res['subject_hr_mae']:.2f} | SNR={res['subject_snr_db']:.2f} dB",
                flush=True,
            )

            # Save incremental progress safely after every run
            pd.DataFrame(abl_rows).to_csv(os.path.join(out_dir, "exp_ablation_multiseed_per_seed.csv"), index=False)
            pd.DataFrame(strata_rows).to_csv(os.path.join(out_dir, "exp_motion_strata_multiseed_per_seed.csv"), index=False)
            pd.DataFrame(cross_rows).to_csv(os.path.join(out_dir, "exp_ablation_cross_per_seed.csv"), index=False)
            pd.DataFrame(cond_rows).to_csv(os.path.join(out_dir, "exp_ablation_cross_cond_per_seed.csv"), index=False)

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        del train_loader, val_loader, test_loader
        gc.collect()

    # Summarize all tables across seeds
    df_abl = pd.DataFrame(abl_rows)
    abl_sum = []
    for variant, grp in df_abl.groupby("Variant", sort=False):
        fmt = lambda c, d=2: f"{grp[c].mean():.{d}f} ± {grp[c].std(ddof=0):.{d}f}"
        abl_sum.append({
            "Variant": variant,
            "Gauss_Filter": grp["Gauss_Filter"].iloc[0],
            "Lambda_Smooth": grp["Lambda_Smooth"].iloc[0],
            "Val_Pearson": fmt("Val_Pearson", 4),
            "Subj_Pearson": fmt("Subj_Pearson", 4),
            "Subj_HR_MAE": fmt("Subj_HR_MAE", 2),
            "Subj_HR_RMSE": fmt("Subj_HR_RMSE", 2),
            "Subj_SNR_dB": fmt("Subj_SNR_dB", 2),
            "Sensor_HR_MAE": fmt("Sensor_HR_MAE", 2),
            "Val_Mean": float(grp["Val_Pearson"].mean()),
            "Test_Mean": float(grp["Subj_Pearson"].mean()),
        })
    df_abl_sum = pd.DataFrame(abl_sum)
    df_abl_sum.to_csv(os.path.join(out_dir, "exp_ablation_multiseed.csv"), index=False)
    with open(os.path.join(out_dir, "exp_ablation_multiseed.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-Seed Ablation Study with Validation-Pearson Selection (Seeds 42, 43, 44)\n\n")
        f.write(df_abl_sum.drop(columns=["Val_Mean", "Test_Mean"]).to_markdown(index=False))
        f.write("\n")

    # 2x3 Factorial view
    df_2x3 = df_abl_sum[df_abl_sum["Gauss_Filter"].isin(["Off", "On"])].copy()
    df_2x3.to_csv(os.path.join(out_dir, "exp_ablation_2x2.csv"), index=False)
    with open(os.path.join(out_dir, "exp_ablation_2x2.md"), "w", encoding="utf-8") as f:
        f.write("# 2x3 Factorial Ablation: Gaussian Logit Filter x Smoothness Loss (Seeds 42, 43, 44)\n\n")
        f.write(df_2x3.drop(columns=["Val_Mean", "Test_Mean"]).to_markdown(index=False))
        f.write("\n")

    # Motion strata summary
    df_st = pd.DataFrame(strata_rows)
    st_sum = []
    for (variant, stratum), grp in df_st.groupby(["Variant", "Stratum"], sort=False):
        fmt = lambda c, d=2: f"{grp[c].mean():.{d}f} ± {grp[c].std(ddof=0):.{d}f}"
        st_sum.append({
            "Variant": variant,
            "Stratum": stratum,
            "Subjects": int(grp["Subjects"].iloc[0]),
            "Mean_Pearson": fmt("Mean_Pearson", 4),
            "Mean_HR_MAE": fmt("Mean_HR_MAE", 2),
            "Mean_SNR_dB": fmt("Mean_SNR_dB", 2),
        })
    df_st_sum = pd.DataFrame(st_sum)
    df_st_sum.to_csv(os.path.join(out_dir, "exp_motion_strata_multiseed.csv"), index=False)
    with open(os.path.join(out_dir, "exp_motion_strata_multiseed.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-Seed Mouth-Motion Stratified Evaluation (UBFC Test Split, Seeds 42, 43, 44)\n\n")
        f.write(df_st_sum.to_markdown(index=False))
        f.write("\n")

    # Cross-dataset ablation summary
    df_cr = pd.DataFrame(cross_rows)
    cr_sum = []
    for (ds_name, variant), grp in df_cr.groupby(["Dataset", "Variant"], sort=False):
        fmt = lambda c, d=2: f"{grp[c].mean():.{d}f} ± {grp[c].std(ddof=0):.{d}f}"
        cr_sum.append({
            "Dataset": ds_name,
            "Variant": variant,
            "Sessions": int(grp["Sessions"].iloc[0]),
            "Subj_Pearson": fmt("Subj_Pearson", 4),
            "Oracle_Lag_Pearson": fmt("Oracle_Lag_Pearson", 4),
            "Subj_HR_MAE": fmt("Subj_HR_MAE", 2),
            "Subj_SNR_dB": fmt("Subj_SNR_dB", 2),
        })
    df_cr_sum = pd.DataFrame(cr_sum)
    df_cr_sum.to_csv(os.path.join(out_dir, "exp_ablation_cross_dataset.csv"), index=False)
    with open(os.path.join(out_dir, "exp_ablation_cross_dataset.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-Seed Ablation Cross-Dataset Generalization (MCD-rPPG & MMPD)\n\n")
        f.write(df_cr_sum.to_markdown(index=False))
        f.write("\n")

    df_cd = pd.DataFrame(cond_rows)
    cd_sum = []
    for (ds_name, cond_label, variant), grp in df_cd.groupby(["Dataset", "Condition", "Variant"], sort=False):
        fmt = lambda c, d=2: f"{grp[c].mean():.{d}f} ± {grp[c].std(ddof=0):.{d}f}"
        cd_sum.append({
            "Dataset": ds_name,
            "Condition": cond_label,
            "Variant": variant,
            "Sessions": int(grp["Sessions"].iloc[0]),
            "Mean_Pearson": fmt("Mean_Pearson", 4),
            "Oracle_Lag_Pearson": fmt("Oracle_Lag_Pearson", 4),
            "Mean_HR_MAE": fmt("Mean_HR_MAE", 2),
            "Mean_SNR_dB": fmt("Mean_SNR_dB", 2),
        })
    df_cd_sum = pd.DataFrame(cd_sum)
    df_cd_sum.to_csv(os.path.join(out_dir, "exp_ablation_cross_conditions.csv"), index=False)
    with open(os.path.join(out_dir, "exp_ablation_cross_conditions.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-Seed Ablation Condition-Stratified Cross-Dataset Robustness\n\n")
        f.write(df_cd_sum.to_markdown(index=False))
        f.write("\n")

    print("\n===== 3-SEED ABLATION SUMMARY =====", flush=True)
    print(df_abl_sum.to_string(index=False), flush=True)
    print("\n===== 3-SEED ABLATION CROSS-DATASET =====", flush=True)
    print(df_cr_sum.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=os.path.join(REPO_ROOT, "data", "UBFC-rPPG", "1"))
    parser.add_argument("--mmpd-root", type=str, default=os.path.join(REPO_ROOT, "data", "MMPD", "1"))
    parser.add_argument("--mcd-root", type=str, default=os.path.join(REPO_ROOT, "data", "MCD-rPPG", "1"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(REPO_ROOT, "results", "ubfc_full"))
    parser.add_argument("--ckpt-dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "ubfc_full"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--parts", type=str, nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if "A" in args.parts:
        run_part_a(args.data_root, args.out_dir, args.ckpt_dir, args.seeds, args.mmpd_root, args.mcd_root)
    if "B" in args.parts:
        run_part_b(args.data_root, args.out_dir, args.ckpt_dir, args.seeds)
    if "C" in args.parts:
        run_part_c(
            args.data_root, args.out_dir, args.ckpt_dir, args.seeds,
            args.mmpd_root, args.mcd_root, epochs=args.epochs,
        )
    print("\nALL REVIEWER REVISION EXPERIMENTS COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
