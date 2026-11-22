"""Standardized physiological signal processing, HR estimation, and statistical metrics."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import ttest_rel, wilcoxon


def bandpass_filter(
    signal: np.ndarray,
    fs: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter safe for short clips."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 16:
        return x.astype(np.float32)

    nyq = 0.5 * fs
    low = max(1e-4, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    if low >= high:
        return x.astype(np.float32)

    b, a = butter(order, [low, high], btype="band")
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    filtered = filtfilt(b, a, x, padlen=padlen)
    return filtered.astype(np.float32)


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """Safe Pearson correlation returning 0.0 for degenerate constant signals."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    p = p - p.mean()
    t = t - t.mean()
    denom = np.sqrt(np.sum(p * p) * np.sum(t * t))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.sum(p * t) / denom, -1.0, 1.0))


def _parabolic_peak(mag: np.ndarray, freqs: np.ndarray, idx: int) -> float:
    """Quadratic interpolation around discrete FFT peak for sub-bin frequency accuracy."""
    if idx <= 0 or idx >= len(mag) - 1:
        return float(freqs[idx])
    y0, y1, y2 = float(mag[idx - 1]), float(mag[idx]), float(mag[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[idx])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    df = float(freqs[1] - freqs[0])
    return float(freqs[idx] + delta * df)


def estimate_hr_fft(
    signal: np.ndarray,
    fs: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    nfft: int = 4096,
) -> float:
    """
    Estimate heart rate (bpm) using Hann-windowed, zero-padded FFT + parabolic interpolation.

    With fs=30 Hz and nfft=4096, frequency resolution is 0.44 bpm (vs. 14.06 bpm in raw 128-pt FFT).
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return 0.5 * (low_hz + high_hz) * 60.0

    win = np.hanning(len(x))
    x_win = x * win
    nfft_eff = max(int(nfft), len(x))
    freqs = np.fft.rfftfreq(nfft_eff, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x_win, n=nfft_eff))

    valid_idx = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
    if len(valid_idx) == 0:
        return 0.5 * (low_hz + high_hz) * 60.0

    local_max = int(np.argmax(mag[valid_idx]))
    peak_idx = int(valid_idx[local_max])
    # Only interpolate inside valid bounds
    if valid_idx[0] < peak_idx < valid_idx[-1]:
        peak_freq = _parabolic_peak(mag, freqs, peak_idx)
    else:
        peak_freq = float(freqs[peak_idx])
    return float(peak_freq * 60.0)


def estimate_hr_welch(
    signal: np.ndarray,
    fs: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    nfft: int = 4096,
) -> float:
    """Estimate heart rate (bpm) using Welch PSD (recommended for long subject-level traces)."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    if len(x) < 32 or np.std(x) < 1e-12:
        return estimate_hr_fft(x, fs=fs, low_hz=low_hz, high_hz=high_hz, nfft=nfft)

    nperseg = min(len(x), int(round(fs * 8)))
    noverlap = nperseg // 2
    freqs, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=max(nfft, nperseg))
    valid_idx = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
    if len(valid_idx) == 0:
        return 0.5 * (low_hz + high_hz) * 60.0
    local_max = int(np.argmax(pxx[valid_idx]))
    peak_idx = int(valid_idx[local_max])
    return float(freqs[peak_idx] * 60.0)


def compute_bvp_snr(
    signal: np.ndarray,
    gt_hr_bpm: float,
    fs: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    nfft: int = 4096,
    bw_hz: float = 0.1,
) -> float:
    """
    Compute physiological BVP Signal-to-Noise Ratio (dB) following De Haan & Jeanne (2013).
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return -30.0

    win = np.hanning(len(x))
    nfft_eff = max(int(nfft), len(x))
    freqs = np.fft.rfftfreq(nfft_eff, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x * win, n=nfft_eff)) ** 2

    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    f_hr = gt_hr_bpm / 60.0
    # Respect Rayleigh mainlobe width for short clips while matching standard 0.1 Hz on >=15s traces
    bw_eff = max(bw_hz, 1.5 * fs / len(x))
    sig_mask = in_band & (
        (np.abs(freqs - f_hr) <= bw_eff) | (np.abs(freqs - 2.0 * f_hr) <= 2.0 * bw_eff)
    )
    noise_mask = in_band & (~sig_mask)

    sig_power = float(np.sum(power[sig_mask]))
    noise_power = float(np.sum(power[noise_mask]))
    if noise_power < 1e-12:
        return 30.0
    if sig_power < 1e-12:
        return -30.0
    return float(10.0 * np.log10(sig_power / noise_power))


def stitch_subject_predictions(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
    subjects: List[str],
    starts: List[int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Overlap-add clip predictions back into continuous per-subject waveforms.
    """
    by_subj = defaultdict(list)
    for p, t, s, st in zip(preds, targets, subjects, starts):
        by_subj[s].append((int(st), np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)))

    stitched: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for subj, items in by_subj.items():
        max_end = max(st + len(p) for st, p, _ in items)
        p_acc = np.zeros(max_end, dtype=np.float64)
        t_acc = np.zeros(max_end, dtype=np.float64)
        w_acc = np.zeros(max_end, dtype=np.float64)

        for st, p, t in items:
            L = len(p)
            # Standardize each clip prediction before overlap-add so amplitude scaling is uniform
            p_std = np.std(p)
            p_norm = (p - np.mean(p)) / (p_std if p_std > 1e-8 else 1.0)
            t_std = np.std(t)
            t_norm = (t - np.mean(t)) / (t_std if t_std > 1e-8 else 1.0)
            win = np.hanning(L) + 1e-3
            p_acc[st:st + L] += p_norm * win
            t_acc[st:st + L] += t_norm * win
            w_acc[st:st + L] += win

        valid = w_acc > 0
        p_full = np.zeros(max_end, dtype=np.float32)
        t_full = np.zeros(max_end, dtype=np.float32)
        p_full[valid] = (p_acc[valid] / w_acc[valid]).astype(np.float32)
        t_full[valid] = (t_acc[valid] / w_acc[valid]).astype(np.float32)
        # Trim unpredicted trailing/leading zeros if any
        nz = np.where(valid)[0]
        if len(nz) > 0:
            lo, hi = int(nz[0]), int(nz[-1]) + 1
            stitched[subj] = (p_full[lo:hi], t_full[lo:hi])
    return stitched


def evaluate_predictions(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
    subjects: Optional[List[str]] = None,
    starts: Optional[List[int]] = None,
    fs: float = 30.0,
    hr_method: str = "fft",
) -> Dict[str, object]:
    """
    Evaluate clip-level and (if metadata provided) subject-level rPPG metrics.
    """
    hr_fn = estimate_hr_welch if hr_method == "welch" else estimate_hr_fft

    clip_corrs, clip_abs_errs, clip_sq_errs, clip_snrs = [], [], [], []
    for p_raw, t_raw in zip(preds, targets):
        p = bandpass_filter(p_raw, fs=fs)
        t = bandpass_filter(t_raw, fs=fs)
        clip_corrs.append(pearson_corr(p, t))
        bpm_p = estimate_hr_fft(p, fs=fs)
        bpm_t = estimate_hr_fft(t, fs=fs)
        err = abs(bpm_p - bpm_t)
        clip_abs_errs.append(err)
        clip_sq_errs.append(err ** 2)
        clip_snrs.append(compute_bvp_snr(p, bpm_t, fs=fs))

    out: Dict[str, object] = {
        "clip_pearson": float(np.mean(clip_corrs)) if clip_corrs else 0.0,
        "clip_hr_mae": float(np.mean(clip_abs_errs)) if clip_abs_errs else 0.0,
        "clip_hr_rmse": float(np.sqrt(np.mean(clip_sq_errs))) if clip_sq_errs else 0.0,
        "clip_snr_db": float(np.mean(clip_snrs)) if clip_snrs else 0.0,
        "clip_corrs_list": clip_corrs,
        "clip_abs_errs_list": clip_abs_errs,
    }

    if subjects is not None and starts is not None and len(subjects) == len(preds):
        stitched = stitch_subject_predictions(preds, targets, subjects, starts)
        subj_metrics = {}
        s_corrs, s_maes, s_sq, s_snrs = [], [], [], []
        for subj, (p_full, t_full) in sorted(stitched.items()):
            p = bandpass_filter(p_full, fs=fs)
            t = bandpass_filter(t_full, fs=fs)
            c = pearson_corr(p, t)
            bpm_p = hr_fn(p, fs=fs)
            bpm_t = hr_fn(t, fs=fs)
            err = abs(bpm_p - bpm_t)
            snr = compute_bvp_snr(p, bpm_t, fs=fs)
            s_corrs.append(c)
            s_maes.append(err)
            s_sq.append(err ** 2)
            s_snrs.append(snr)
            subj_metrics[subj] = {
                "pearson": c,
                "hr_pred": bpm_p,
                "hr_gt": bpm_t,
                "hr_mae": err,
                "snr_db": snr,
            }
        out.update(
            {
                "subject_pearson": float(np.mean(s_corrs)) if s_corrs else 0.0,
                "subject_hr_mae": float(np.mean(s_maes)) if s_maes else 0.0,
                "subject_hr_rmse": float(np.sqrt(np.mean(s_sq))) if s_sq else 0.0,
                "subject_snr_db": float(np.mean(s_snrs)) if s_snrs else 0.0,
                "per_subject": subj_metrics,
            }
        )
    return out


def paired_significance_tests(
    scores_method: np.ndarray,
    scores_baseline: np.ndarray,
    higher_is_better: bool = True,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute paired Wilcoxon signed-rank, paired t-test, winrate, and 95% bootstrap CI.
    """
    a = np.asarray(scores_method, dtype=np.float64)
    b = np.asarray(scores_baseline, dtype=np.float64)
    diff = (a - b) if higher_is_better else (b - a)

    winrate = float(np.mean(diff > 0) * 100.0)
    mean_gain = float(np.mean(diff))

    if len(diff) >= 2 and np.any(diff != 0):
        try:
            _, w_p = wilcoxon(diff)
        except Exception:
            w_p = 1.0
        try:
            _, t_p = ttest_rel(a, b)
        except Exception:
            t_p = 1.0
    else:
        w_p, t_p = 1.0, 1.0

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(diff)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(float(np.mean(diff[idx])))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_gain": mean_gain,
        "winrate_pct": winrate,
        "wilcoxon_p": float(w_p),
        "ttest_p": float(t_p),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }
