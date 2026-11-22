"""Lightweight unit sanity tests for SKIA-v2 components."""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baselines_classical import CLASSICAL_METHODS
from baselines_deep import build_deep_model
from metrics import compute_bvp_snr, estimate_hr_fft, estimate_hr_welch, pearson_corr
from roi_extractor import EXPERIMENT_A_VARIANTS, build_variant_masks, make_geometric_region_masks
from skia_v2_model import SKIANetV2, SKIAv2Loss
from splits import get_kfold_splits, get_subject_split


def test_splits_leak_free():
    subjs = [f"subject{i}" for i in range(1, 43)]
    sp = get_subject_split(subjs, seed=42, preserve_v1_test=False)
    assert set(sp["train"]).isdisjoint(sp["val"])
    assert set(sp["train"]).isdisjoint(sp["test"])
    assert set(sp["val"]).isdisjoint(sp["test"])
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == 42

    folds = get_kfold_splits(subjs, n_splits=5, seed=42)
    assert len(folds) == 5
    for f in folds:
        assert set(f["train"]).isdisjoint(f["val"])
        assert set(f["train"]).isdisjoint(f["test"])
        assert set(f["val"]).isdisjoint(f["test"])


def test_hr_estimation_precision():
    fs = 30.0
    t = np.arange(128) / fs
    true_bpm = 77.3
    sig = np.sin(2 * np.pi * (true_bpm / 60.0) * t)
    est = estimate_hr_fft(sig, fs=fs, nfft=4096)
    assert abs(est - true_bpm) < 1.5, f"Expected ~{true_bpm}, got {est}"
    assert pearson_corr(sig, sig) > 0.999
    assert compute_bvp_snr(sig, true_bpm, fs=fs) > 5.0


def test_models_forward_and_variance():
    B, T, H, W = 2, 64, 64, 64
    x = torch.randn(B, 3, T, H, W)
    m = torch.ones(B, 4, T, H, W)
    y = torch.randn(B, T)

    for name in ("physnet", "tscan", "efficientphys", "physformer", "skia_raw", "skia_full"):
        net = build_deep_model(name)
        out = net(x)
        assert out.shape == (B, T), f"{name} shape mismatch: {out.shape}"
        assert float(out.std().item()) > 0.5, f"{name} variance collapsed: {out.std()}"

    v2 = SKIANetV2(num_regions=4, gate_mode="stable_dynamic")
    out, w = v2(x, m, return_weights=True)
    assert out.shape == (B, T)
    assert w.shape == (B, 4, T)
    assert torch.allclose(w.sum(dim=1), torch.ones(B, T), atol=1e-4)
    loss_fn = SKIAv2Loss()
    loss, stats = loss_fn(out, y, w)
    loss.backward()
    assert np.isfinite(stats["loss_total"])


def test_classical_and_variants():
    T, H, W = 180, 64, 64
    crops = np.random.uniform(0.4, 0.8, size=(T, H, W, 3)).astype(np.float32)
    full = np.random.uniform(0.2, 0.9, size=(T, 96, 128, 3)).astype(np.float32)
    rm = np.broadcast_to(make_geometric_region_masks(H, W)[None], (T, 4, H, W)).copy()
    sm = np.ones((T, H, W), dtype=np.float32)

    for v in EXPERIMENT_A_VARIANTS:
        vid, m = build_variant_masks(v, full, crops, rm, sm)
        assert m is None or m.shape == vid.shape[1:3] or m.shape == vid.shape[:3], f"{v} mask shape {None if m is None else m.shape}"

    t = np.arange(T) / 30.0
    r = 0.7 + 0.01 * np.sin(2 * np.pi * 1.2 * t)
    g = 0.5 + 0.02 * np.sin(2 * np.pi * 1.2 * t)
    b = 0.4 + 0.008 * np.sin(2 * np.pi * 1.2 * t)
    for name, fn in CLASSICAL_METHODS.items():
        out = fn(r, g, b, 30.0)
        assert len(out) == T, f"{name} length mismatch"


if __name__ == "__main__":
    test_splits_leak_free()
    test_hr_estimation_precision()
    test_models_forward_and_variance()
    test_classical_and_variants()
    print("ALL SANITY TESTS PASSED!")
