"""Classical signal-processing rPPG baselines: GREEN, PCA, ICA, CHROM, POS, and LGI."""

from typing import Callable, Dict, Optional, Tuple
import numpy as np
from sklearn.decomposition import FastICA

from metrics import bandpass_filter


def extract_rgb_traces(
    frames: np.ndarray,
    masks: Optional[np.ndarray] = None,
    max_frames: Optional[int] = None,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract spatially averaged R, G, B temporal traces over an optional (H,W) or (T,H,W) mask.
    """
    T = len(frames) if max_frames is None else min(len(frames), int(max_frames))
    r_list, g_list, b_list = [], [], []

    static_mask = masks is not None and masks.ndim == 2
    if static_mask:
        m2 = np.asarray(masks, dtype=np.float32)
        denom_static = float(m2.sum()) + 1e-8

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        x = np.asarray(frames[start:end], dtype=np.float32)

        if masks is None:
            r = x[..., 0].mean(axis=(1, 2))
            g = x[..., 1].mean(axis=(1, 2))
            b = x[..., 2].mean(axis=(1, 2))
        elif static_mask:
            r = (x[..., 0] * m2).sum(axis=(1, 2)) / denom_static
            g = (x[..., 1] * m2).sum(axis=(1, 2)) / denom_static
            b = (x[..., 2] * m2).sum(axis=(1, 2)) / denom_static
        else:
            m = np.asarray(masks[start:end], dtype=np.float32)
            denom = m.sum(axis=(1, 2)) + 1e-8
            r = (x[..., 0] * m).sum(axis=(1, 2)) / denom
            g = (x[..., 1] * m).sum(axis=(1, 2)) / denom
            b = (x[..., 2] * m).sum(axis=(1, 2)) / denom

        r_list.append(r)
        g_list.append(g)
        b_list.append(b)

    return np.concatenate(r_list), np.concatenate(g_list), np.concatenate(b_list)


def _spectral_peak_ratio(sig: np.ndarray, fs: float = 30.0, low: float = 0.7, high: float = 2.5) -> float:
    x = sig - np.mean(sig)
    if np.std(x) < 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(max(1024, len(x)), d=1.0 / fs)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=max(1024, len(x)))) ** 2
    band = (freqs >= low) & (freqs <= high)
    total = float(np.sum(power[band])) + 1e-12
    return float(np.max(power[band]) / total)


def run_green(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Green channel rPPG (Verkruysse et al., 2008)."""
    gn = g / (np.mean(g) + 1e-8) - 1.0
    return bandpass_filter(-gn, fs=fs)


def run_chrom(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Windowed CHROM algorithm (De Haan & Jeanne, IEEE TBME 2013) with overlap-add.
    """
    rgb = np.stack([r, g, b], axis=1).astype(np.float64)
    T = len(rgb)
    win_len = max(16, int(round(1.6 * fs)))
    if win_len % 2 == 1:
        win_len += 1
    step = max(1, win_len // 2)

    if T <= win_len:
        mean_rgb = np.mean(rgb, axis=0) + 1e-8
        rn, gn, bn = (rgb / mean_rgb).T
        xs = 3.0 * rn - 2.0 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        xf = bandpass_filter(xs, fs=fs)
        yf = bandpass_filter(ys, fs=fs)
        alpha = np.std(xf) / (np.std(yf) + 1e-8)
        return (xf - alpha * yf).astype(np.float32)

    S = np.zeros(T, dtype=np.float64)
    W = np.zeros(T, dtype=np.float64)
    hann = np.hanning(win_len)
    for start in range(0, T - win_len + 1, step):
        seg = rgb[start:start + win_len]
        seg_mean = np.mean(seg, axis=0) + 1e-8
        rn, gn, bn = (seg / seg_mean - 1.0).T
        xs = 3.0 * rn - 2.0 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        xf = bandpass_filter(xs, fs=fs)
        yf = bandpass_filter(ys, fs=fs)
        alpha = np.std(xf) / (np.std(yf) + 1e-8)
        s_win = (xf - alpha * yf) * hann
        S[start:start + win_len] += s_win
        W[start:start + win_len] += hann

    valid = W > 1e-8
    S[valid] /= W[valid]
    return bandpass_filter(S, fs=fs)


def run_pos(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., IEEE TBME 2016).
    """
    rgb = np.stack([r, g, b], axis=1).astype(np.float64)
    T = len(rgb)
    win_len = max(16, int(round(1.6 * fs)))
    H_out = np.zeros(T, dtype=np.float64)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)

    for n in range(0, T):
        m = n - win_len + 1
        if m >= 0:
            cn = rgb[m:n + 1]
            cn_mean = np.mean(cn, axis=0) + 1e-8
            cn_norm = cn / cn_mean
            s = proj @ cn_norm.T  # (2, L)
            alpha = np.std(s[0]) / (np.std(s[1]) + 1e-8)
            h_seg = s[0] + alpha * s[1]
            h_seg = h_seg - np.mean(h_seg)
            H_out[m:n + 1] += h_seg

    # Beer-Lambert convention: systolic BVP increase lowers reflected green/chrominance -> invert sign
    return bandpass_filter(-H_out, fs=fs)


def run_pca(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """PCA-based blind source separation picking the component with cleanest cardiac peak."""
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    comps = X @ vt.T
    best = None
    best_score = -1.0
    for k in range(comps.shape[1]):
        cand = bandpass_filter(comps[:, k], fs=fs)
        sc = _spectral_peak_ratio(cand, fs=fs)
        if sc > best_score:
            best_score = sc
            best = cand
    return best if best is not None else bandpass_filter(g, fs=fs)


def run_ica(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """FastICA-based rPPG (Poh et al., 2010) with spectral component selection."""
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    try:
        ica = FastICA(n_components=3, max_iter=500, tol=1e-3, random_state=42, whiten="unit-variance")
        comps = ica.fit_transform(X)
    except Exception:
        return run_pca(r, g, b, fs=fs)

    best = None
    best_score = -1.0
    for k in range(comps.shape[1]):
        cand = bandpass_filter(comps[:, k], fs=fs)
        sc = _spectral_peak_ratio(cand, fs=fs)
        if sc > best_score:
            best_score = sc
            best = cand
    return best if best is not None else bandpass_filter(g, fs=fs)


def run_lgi(r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """Local Group Invariance (Pilz et al., CVPRW 2018)."""
    X = np.stack([r, g, b], axis=1).astype(np.float64)
    X_norm = X / (np.mean(X, axis=0) + 1e-8) - 1.0
    u, _, _ = np.linalg.svd(X_norm.T, full_matrices=False)
    S = u[:, 0:1]
    P = np.eye(3) - S @ S.T
    Y = X_norm @ P.T
    sig = Y[:, 1] - Y[:, 0]
    return bandpass_filter(sig, fs=fs)


CLASSICAL_METHODS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]] = {
    "GREEN": run_green,
    "PCA": run_pca,
    "ICA": run_ica,
    "CHROM": run_chrom,
    "POS": run_pos,
    "LGI": run_lgi,
}
