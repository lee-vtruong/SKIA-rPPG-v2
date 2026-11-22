"""Automated dataset download & setup instructions for UBFC-rPPG, PURE, UBFC-Phys, and MMPD."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from typing import List, Tuple

import cv2
import numpy as np
import requests
import scipy.io
from scipy.signal import resample
from huggingface_hub import HfApi, hf_hub_url


# Deduplicated official UBFC-rPPG DATASET_2 Google Drive folder IDs (42 subjects)
UBFC_GDRIVE_FOLDERS = [
    "1q6U7jKRBUtr6N2rVs1vwZ2wuUT2VDoRy",
    "1tJdI-E143hW0q0NHxmPPkLcr0UvZgeNz",
    "1vjXMUmi9CDWvAnrLNoteFNMoxU48Dhzt",
    "1y9wb_8Vn1bt34gSMBuc7ux987AzpNEeg",
    "1yK1wZdt33J6OxBSyv8kRtNieaH0SgSpi",
    "1yR3hc7nfVqJt8VsXz9AIM1r2R5H-pdGC",
    "1qCvEJjNRfMVuz4VoQgbDS-RMvgh5Hn9a",
    "1qRx5WQvLOwiHM_4w09aKqSLeWVbG4qHE",
    "1qavAF-3T7IQdnytHSxqCesdfGqzgavlJ",
    "1ql-e1MDCCr4VFLEPPbMpAOXPbYCIX95f",
    "1rCP6v2FotK0-SVJGSNgjeA6ppbBb3z0L",
    "1rGs5PpY62WdHet_PH6fBGqTMVJvxGAzt",
    "1rgia82iJ_6igB0U3tj522jxDa0pdqZTz",
    "1rjLl2Nm5ZsDLAsSCCYPQCf08HV5JJwYX",
    "1rrd41TEOi9H1_k0cV8pQFZURUE5yS5pb",
    "1s46xn-OMg1pEVXnMvl3yB1bJfwBmMX-g",
    "1sJXdmzk9mTG_kpCjo00lPkcDKyo5Aqhs",
    "1sWjC0FyshavhK8ztwg4I95bF4QU6nybu",
    "1soR7P6uArXdJEeaxER7bBKTPuzlhsq1b",
    "1szbMEUYHPF_VJjGtdtgUjo4NL7OPKSZ7",
    "1t2usXJ0VjLnM3wV7raDUnVGc4Z_eCz3V",
    "1tAyikar9Zeu6jjJZmx5m7LcyxIeq7XKX",
    "1tWXN4qYLmzHa5_hx_elqU_sgl1aYy0vO",
    "1twV7BMwxwF14XGZfxzkS-9P4ezfZty1P",
    "1u3oijGKzB0G_5o1Ju2knvn2KGUtEwOEk",
    "1uMT6Cv-JnlXhOVoPZ8nxPtqPQYy6Gdva",
    "1uOfj5OY22f1vMo_Gc5no9H7Nbb6U3dXK",
    "1uggXwmu9pOHOS1Ds2CO4pEgfOZoVT57j",
    "1v5hfhYpjBsUDmrQVpaDEUZmtWZw06etA",
    "1vEDM3WM0vbaiWdiZcIRbtE9a0plR2Tsc",
    "1vLl3Sd3Vfrml6BwXuMAqzYDB82v27pBv",
    "1vb1dYR72dGH0pcr-LVhtJSKIQJb7GOep",
    "1w2In1mSSLtHqOMPwnsepOQ9n6gzYJKPd",
    "1wHgO1SZw9dI4MOlslapzSXr4FO6xcowj",
    "1wSYMwaW7l4wUe5OCF37CgsC4_T4YM6qE",
    "1wf6sjVYvYoJaUmK0PJa11y7TKQEd2wRd",
    "1x3OJT8ArvFpso54_EK0DNPpy2aSJ_UTL",
    "1x8szB8zXsqAQgHnXKQWOUdhtkYiGmN_J",
    "1xH-vlaB5aggjVVlm-rvydVOfoJWn7YwO",
    "1xSXY4if2UbdoBw47FuUM8LOdKzavwIkY",
    "1xxOONETbO_zyFq7TMcOdLlvc8vDuu0Aa",
    "1y3az0As2ePyGGwT4OzGwmafAVw3EdJqk",
]


def _http_get_range_with_retry(url: str, start: int, end: int, retries: int = 4) -> bytes:
    """Fetch inclusive byte range [start, end] with retries."""
    headers = {"Range": f"bytes={start}-{end}"}
    expected = end - start + 1
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 206 and len(r.content) == expected:
                return r.content
            if r.status_code == 200 and len(r.content) >= end + 1:
                return r.content[start:end + 1]
        except Exception as exc:
            last_exc = exc
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Range fetch failed [{start}-{end}] from {url}: {last_exc}")


def _parallel_download_range(url: str, start: int, total_bytes: int, workers: int = 6) -> bytes:
    """Download [start, start + total_bytes) in parallel chunks."""
    chunk_sz = max(4 * 1024 * 1024, (total_bytes + workers - 1) // workers)
    tasks: List[Tuple[int, int, int]] = []
    pos = 0
    while pos < total_bytes:
        s = start + pos
        e = min(start + total_bytes - 1, s + chunk_sz - 1)
        tasks.append((pos, s, e))
        pos += (e - s + 1)

    buf = bytearray(total_bytes)

    def _worker(item: Tuple[int, int, int]) -> None:
        rel_off, s, e = item
        data = _http_get_range_with_retry(url, s, e)
        expected = e - s + 1
        if len(data) != expected:
            data = data[:expected]
        buf[rel_off:rel_off + len(data)] = data

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_worker, tasks))
    return bytes(buf)


def _decode_raw_avi_prefix(
    vid_url: str,
    bvp_len: int,
    max_frames: int,
    start_frame: int = 0,
) -> np.ndarray:
    """Stream frames [start_frame, end_frame) RGB from an uncompressed RIFF AVI via HTTP Range."""
    r0 = requests.get(vid_url, stream=True, timeout=20)
    r0.raise_for_status()
    cdn_url = r0.url
    file_size = int(r0.headers.get("Content-Length", "0") or 0)
    r0.close()

    hdr = _http_get_range_with_retry(cdn_url, 0, min(file_size - 1, 262143) if file_size else 262143)
    strf_pos = hdr.find(b"strf")
    movi_pos = hdr.find(b"movi")

    if strf_pos >= 0 and movi_pos >= 0 and movi_pos + 12 < len(hdr):
        _, width, height_signed, _, bpp, comp, _ = struct.unpack(
            "<IiiHHII", hdr[strf_pos + 8:strf_pos + 32]
        )
        height = abs(height_signed)
        bottom_up = height_signed > 0
        chunk_len = int.from_bytes(hdr[movi_pos + 8:movi_pos + 12], "little")
        expected_frame_bytes = width * height * (bpp // 8)
        if comp == 0 and bpp == 24 and chunk_len == expected_frame_bytes:
            movi_start = movi_pos + 4
            padded_chunk = chunk_len + (chunk_len % 2)
            stride = 8 + padded_chunk
            avail_frames = (file_size - movi_start) // stride if file_size else bvp_len
            end_frame = min(max_frames if max_frames > 0 else bvp_len, bvp_len, avail_frames)
            if end_frame <= start_frame:
                return np.zeros((0, height, width, 3), dtype=np.uint8)
            count = end_frame - start_frame
            payload = _parallel_download_range(
                cdn_url, movi_start + start_frame * stride, count * stride, workers=6
            )
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(count, stride)
            pixels = arr[:, 8:8 + chunk_len].reshape(count, height, width, 3)
            if bottom_up:
                return np.ascontiguousarray(pixels[:, ::-1, :, ::-1])
            return np.ascontiguousarray(pixels[:, :, :, ::-1])

    # Generic fallback for compressed/irregular AVI
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tf:
        tmp_path = tf.name
    try:
        with requests.get(cdn_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for ch in r.iter_content(chunk_size=1024 * 1024):
                    if ch:
                        f.write(ch)
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        limit = max_frames if max_frames > 0 else bvp_len
        idx_f = 0
        while len(frames) < limit - start_frame:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx_f >= start_frame:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx_f += 1
        cap.release()
        return np.asarray(frames, dtype=np.uint8)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _downsample_full_frames(frames_rgb: np.ndarray) -> np.ndarray:
    T, H, W, _ = frames_rgb.shape
    if H == 480 and W == 640:
        return frames_rgb.reshape(T, H // 4, 4, W // 4, 4, 3).mean(axis=(2, 4)).astype(np.uint8)
    return np.asarray(
        [cv2.resize(f, (160, 120), interpolation=cv2.INTER_AREA) for f in frames_rgb],
        dtype=np.uint8,
    )


def download_and_preprocess_ubfc_hf(
    out_dir: str,
    repo_id: str = "thachha901/UBFC",
    max_frames: int = 400,
    size: int = 128,
) -> None:
    """Stream-download and preprocess (or incrementally extend) all 42 real UBFC-rPPG subjects."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from roi_extractor import StabilizedFaceROIExtractor, compute_motion_strata  # noqa: E402

    target_root = os.path.join(out_dir, "UBFC-rPPG", "1")
    os.makedirs(target_root, exist_ok=True)

    api = HfApi()
    tree = list(api.list_repo_tree(repo_id, repo_type="dataset", recursive=True))
    subjects = sorted({f.path.split("/")[1] for f in tree if f.path.startswith("UBFC/subject")})
    if not subjects:
        raise RuntimeError(f"No UBFC subjects discovered in {repo_id}")

    print(f"Found {len(subjects)} real UBFC-rPPG subjects in {repo_id} -> {target_root} (target T={max_frames})", flush=True)
    # Match YuNet + 68-pt LBF backend when extending existing YuNet-preprocessed folders seamlessly
    has_existing = any(
        os.path.exists(os.path.join(target_root, s, "face_crops.npy")) for s in subjects[:3]
    )
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=not has_existing)
    print(
        f"Landmark backend: mp={extractor.use_mediapipe}, yunet={getattr(extractor, '_yunet', None) is not None}, "
        f"facemark68={getattr(extractor, '_facemark', None) is not None}",
        flush=True,
    )

    for idx, subj in enumerate(subjects, start=1):
        subj_dir = os.path.join(target_root, subj)
        os.makedirs(subj_dir, exist_ok=True)
        crop_p = os.path.join(subj_dir, "face_crops.npy")
        mask_p = os.path.join(subj_dir, "region_masks.npy")
        skin_p = os.path.join(subj_dir, "skin_masks.npy")
        frame_p = os.path.join(subj_dir, "frames.npy")
        gt_p = os.path.join(subj_dir, "ground_truth.txt")
        req_files = (crop_p, mask_p, skin_p, frame_p, gt_p)

        t_old = 0
        if all(os.path.exists(p) for p in req_files):
            old_crops = np.load(crop_p, mmap_mode="r")
            t_old = int(old_crops.shape[0])
            del old_crops
            if max_frames > 0 and t_old >= max_frames:
                print(f"[{idx}/{len(subjects)}] {subj}: T={t_old} already >= {max_frames}, skipping.", flush=True)
                continue

        t0 = time.time()
        gt_url = hf_hub_url(repo_id, f"UBFC/{subj}/ground_truth.txt", repo_type="dataset")
        gt_text = requests.get(gt_url, timeout=30).text
        gt_arr = np.loadtxt(gt_text.splitlines())
        bvp_len = gt_arr.shape[1] if gt_arr.ndim == 2 else len(gt_arr)

        if max_frames > 0 and t_old >= bvp_len:
            print(f"[{idx}/{len(subjects)}] {subj}: T={t_old} already at video end ({bvp_len}), skipping.", flush=True)
            continue

        vid_url = hf_hub_url(repo_id, f"UBFC/{subj}/vid.avi", repo_type="dataset")
        frames_new = _decode_raw_avi_prefix(
            vid_url, bvp_len=bvp_len, max_frames=max_frames, start_frame=t_old
        )
        if len(frames_new) == 0:
            print(f"[{idx}/{len(subjects)}] {subj}: no additional frames available beyond T={t_old}.", flush=True)
            continue

        ext = extractor.extract_sequence_crops(frames_new, out_size=size, detect_stride=5)
        crops_u8_new = np.clip(np.rint(ext["crops"] * 255.0), 0, 255).astype(np.uint8)
        rmasks_u8_new = (ext["region_masks"] > 0.5).astype(np.uint8)
        smasks_u8_new = (ext["skin_masks"] > 0.5).astype(np.uint8)
        full_small_new = _downsample_full_frames(frames_new)

        if t_old > 0:
            old_c = np.load(crop_p)
            old_r = np.load(mask_p)
            old_s = np.load(skin_p)
            old_f = np.load(frame_p)
            crops_u8 = np.concatenate([old_c, crops_u8_new], axis=0)
            rmasks_u8 = np.concatenate([old_r, rmasks_u8_new], axis=0)
            smasks_u8 = np.concatenate([old_s, smasks_u8_new], axis=0)
            full_small = np.concatenate([old_f, full_small_new], axis=0)
            del old_c, old_r, old_s, old_f
        else:
            crops_u8, rmasks_u8, smasks_u8, full_small = crops_u8_new, rmasks_u8_new, smasks_u8_new, full_small_new

        T = len(crops_u8)
        if gt_arr.ndim == 2:
            np.savetxt(gt_p, gt_arr[:, :T], fmt="%.8e")
        else:
            np.savetxt(gt_p, gt_arr[:T], fmt="%.8e")

        np.save(crop_p, crops_u8)
        np.save(mask_p, rmasks_u8)
        np.save(skin_p, smasks_u8)
        np.save(frame_p, full_small)

        motion = compute_motion_strata(
            crops_u8.astype(np.float32) / 255.0,
            rmasks_u8.astype(np.float32),
        )
        with open(os.path.join(subj_dir, "motion_stats.json"), "w", encoding="utf-8") as f:
            json.dump(motion, f, indent=2)

        dt = time.time() - t0
        print(
            f"[{idx}/{len(subjects)}] {subj}: T={t_old}->{T} frames, "
            f"mouth_motion={motion['mouth_motion']:.5f}, done in {dt:.1f}s",
            flush=True,
        )


def download_and_preprocess_mmpd_hf(
    out_dir: str,
    max_frames: int = 400,
    size: int = 128,
) -> None:
    """Stream-download and preprocess real MMPD trials across Stationary/Rotation/Talking & LED-low/high."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from roi_extractor import StabilizedFaceROIExtractor, compute_motion_strata  # noqa: E402

    target_root = os.path.join(out_dir, "MMPD", "1")
    os.makedirs(target_root, exist_ok=True)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
    print(
        f"MMPD Landmark backend: mp={extractor.use_mediapipe}, yunet={getattr(extractor, '_yunet', None) is not None}",
        flush=True,
    )

    subjects_cfg = [
        ("thachha901/dataset1", "subject1", "dataset1/subject1/p1"),
        ("thachha901/dataset1", "subject2", "dataset1/subject2/p2"),
        ("thachha901/dataset3", "subject19", "subject19/p19"),
        ("thachha901/dataset3", "subject22", "subject22/p22"),
    ]
    trials = [0, 1, 2, 6]  # Stationary/Rotation/Talking (LED-low) + Talking (LED-high)

    total = len(subjects_cfg) * len(trials)
    step_idx = 0
    for repo_id, subj_name, prefix in subjects_cfg:
        for k in trials:
            step_idx += 1
            mat_path = f"{prefix}_{k}.mat"
            u = hf_hub_url(repo_id, mat_path, repo_type="dataset")
            r0 = requests.get(u, stream=True, timeout=20)
            r0.raise_for_status()
            cdn_url = r0.url
            file_size = int(r0.headers.get("Content-Length", "0") or 0)
            r0.close()

            hdr = _http_get_range_with_retry(cdn_url, 0, 135)
            _, nbytes = struct.unpack("<II", hdr[128:136])
            tail_end = (file_size - 1) if file_size > (136 + nbytes) else (136 + nbytes + 65535)
            tail = _http_get_range_with_retry(cdn_url, 136 + nbytes, tail_end)
            mat = scipy.io.loadmat(io.BytesIO(hdr[:128] + tail))
            motion_lbl = str(mat["motion"][0]).strip()
            light_lbl = str(mat["light"][0]).strip()
            ex_lbl = str(mat["exercise"][0]).strip()
            skin_tone = int(mat["skin_color"][0, 0])
            gt_ppg = np.asarray(mat["GT_ppg"], dtype=np.float32).reshape(-1)

            sess_name = f"{subj_name}_k{k}_{motion_lbl}_{light_lbl}".replace(" ", "")
            sess_dir = os.path.join(target_root, sess_name)
            os.makedirs(sess_dir, exist_ok=True)
            req = ("face_crops.npy", "region_masks.npy", "skin_masks.npy", "frames.npy", "ground_truth.txt", "condition_meta.json")
            if all(os.path.exists(os.path.join(sess_dir, fn)) for fn in req):
                print(f"[{step_idx}/{total}] {sess_name}: already exists, skipping.", flush=True)
                continue

            t0 = time.time()
            comp = _parallel_download_range(cdn_url, 136, nbytes, workers=8)
            raw = zlib.decompress(comp)
            del comp
            dims = struct.unpack("<4i", raw[32:48])
            arr_bytes = int(np.prod(dims)) * 4
            offset = len(raw) - arr_bytes
            arr = np.frombuffer(raw, dtype=np.float32, offset=offset).reshape(dims, order="F")
            T = min(max_frames if max_frames > 0 else dims[0], dims[0], len(gt_ppg))
            frames_rgb = np.clip(arr[:T] * 255.0, 0, 255).astype(np.uint8)
            del arr, raw
            gc.collect()

            ext = extractor.extract_sequence_crops(frames_rgb, out_size=size, detect_stride=3)
            crops_u8 = np.clip(np.rint(ext["crops"] * 255.0), 0, 255).astype(np.uint8)
            rmasks_u8 = (ext["region_masks"] > 0.5).astype(np.uint8)
            smasks_u8 = (ext["skin_masks"] > 0.5).astype(np.uint8)
            full_small = _downsample_full_frames(frames_rgb)

            np.save(os.path.join(sess_dir, "face_crops.npy"), crops_u8)
            np.save(os.path.join(sess_dir, "region_masks.npy"), rmasks_u8)
            np.save(os.path.join(sess_dir, "skin_masks.npy"), smasks_u8)
            np.save(os.path.join(sess_dir, "frames.npy"), full_small)
            np.savetxt(os.path.join(sess_dir, "ground_truth.txt"), gt_ppg[:T], fmt="%.8e")

            motion_stats = compute_motion_strata(ext["crops"], ext["region_masks"])
            meta = {
                "dataset": "MMPD",
                "repo": repo_id,
                "subject": subj_name,
                "trial_k": k,
                "motion": motion_lbl,
                "light": light_lbl,
                "exercise": ex_lbl,
                "skin_color": skin_tone,
                **motion_stats,
            }
            with open(os.path.join(sess_dir, "condition_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            dt = time.time() - t0
            print(
                f"[{step_idx}/{total}] {sess_name}: T={T}, mouth_motion={motion_stats['mouth_motion']:.5f}, "
                f"skin={skin_tone}, done in {dt:.1f}s",
                flush=True,
            )


def download_and_preprocess_mcd_hf(
    out_dir: str,
    repo_id: str = "kyegorov/mcd_rppg",
    max_frames: int = 400,
    size: int = 128,
) -> None:
    """Download and preprocess 10 real MCD-rPPG frontal webcam sessions (5 subjects x Rest/Post-Exercise)."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from roi_extractor import StabilizedFaceROIExtractor, compute_motion_strata  # noqa: E402

    target_root = os.path.join(out_dir, "MCD-rPPG", "1")
    os.makedirs(target_root, exist_ok=True)
    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)

    subj_ids = ["1020", "1024", "1035", "1091", "1097"]
    steps = ["before", "after"]
    total = len(subj_ids) * len(steps)
    idx = 0

    for sid in subj_ids:
        for step in steps:
            idx += 1
            sess_name = f"subj{sid}_{step}"
            sess_dir = os.path.join(target_root, sess_name)
            os.makedirs(sess_dir, exist_ok=True)
            req = ("face_crops.npy", "region_masks.npy", "skin_masks.npy", "frames.npy", "ground_truth.txt", "condition_meta.json")
            if all(os.path.exists(os.path.join(sess_dir, fn)) for fn in req):
                print(f"[{idx}/{total}] {sess_name}: already exists, skipping.", flush=True)
                continue

            t0 = time.time()
            ppg_url = hf_hub_url(repo_id, f"ppg_sync_npy/mcd__{sid}__{step}_front.npy", repo_type="dataset")
            ppg_sync = np.load(io.BytesIO(requests.get(ppg_url, timeout=30).content)).astype(np.float32).reshape(-1)

            vid_url = hf_hub_url(repo_id, f"video/{sid}_FullHDwebcam_{step}.avi", repo_type="dataset")
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tf:
                tmp_path = tf.name
            try:
                with requests.get(vid_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for ch in r.iter_content(chunk_size=1024 * 1024):
                            if ch:
                                f.write(ch)
                cap = cv2.VideoCapture(tmp_path)
                all_frames = []
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    all_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                cap.release()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            n_vid = len(all_frames)
            if len(ppg_sync) != n_vid:
                ppg_sync = resample(ppg_sync, n_vid).astype(np.float32)

            T = min(max_frames if max_frames > 0 else n_vid, n_vid, len(ppg_sync))
            frames_rgb = np.asarray(all_frames[:T], dtype=np.uint8)
            del all_frames

            ext = extractor.extract_sequence_crops(frames_rgb, out_size=size, detect_stride=3)
            crops_u8 = np.clip(np.rint(ext["crops"] * 255.0), 0, 255).astype(np.uint8)
            rmasks_u8 = (ext["region_masks"] > 0.5).astype(np.uint8)
            smasks_u8 = (ext["skin_masks"] > 0.5).astype(np.uint8)
            full_small = _downsample_full_frames(frames_rgb)

            np.save(os.path.join(sess_dir, "face_crops.npy"), crops_u8)
            np.save(os.path.join(sess_dir, "region_masks.npy"), rmasks_u8)
            np.save(os.path.join(sess_dir, "skin_masks.npy"), smasks_u8)
            np.save(os.path.join(sess_dir, "frames.npy"), full_small)
            np.savetxt(os.path.join(sess_dir, "ground_truth.txt"), ppg_sync[:T], fmt="%.8e")

            motion_stats = compute_motion_strata(ext["crops"], ext["region_masks"])
            meta = {
                "dataset": "MCD-rPPG",
                "subject": sid,
                "step": step,
                "orig_frames": n_vid,
                **motion_stats,
            }
            with open(os.path.join(sess_dir, "condition_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            dt = time.time() - t0
            print(
                f"[{idx}/{total}] {sess_name}: T={T}/{n_vid}, mouth_motion={motion_stats['mouth_motion']:.5f}, done in {dt:.1f}s",
                flush=True,
            )


def download_ubfc(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading UBFC-rPPG folders into: {out_dir}")
    for idx, folder_id in enumerate(UBFC_GDRIVE_FOLDERS, start=1):
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"[{idx}/{len(UBFC_GDRIVE_FOLDERS)}] gdown --folder {url}")
        cmd = [sys.executable, "-m", "gdown", "--folder", url, "-O", out_dir, "--remaining-ok"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed on folder {folder_id}: {exc}")
            print("If Google Drive rate-limits gdown, use --source hf or download from:")
            print("  https://sites.google.com/view/ybenezeth/ubfcrppg")


def print_dataset_instructions(dataset: str, out_dir: str) -> None:
    if dataset == "pure":
        print(
            "PURE Dataset Setup:\n"
            "1. Request access at: https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-technical-informatics-and-engineering-informatics/neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure\n"
            f"2. Extract subject folders (e.g., 01-01 .. 10-06) into {out_dir}/PURE/\n"
            "3. Run: python scripts/preprocess_dataset.py --dataset pure --data-root data/PURE --out-dir data/PURE_proc"
        )
    elif dataset == "ubfc_phys":
        print(
            "UBFC-Phys Dataset Setup (includes cognitive/speech tasks T1, T2, T3):\n"
            "1. Download from: https://ieee-dataport.org/open-access/ubfc-phys-2\n"
            f"2. Extract into {out_dir}/UBFC-Phys/\n"
            "3. Convert Task folders to frames.npy + ground_truth.txt, then preprocess facial ROIs with scripts/preprocess_dataset.py"
        )
    elif dataset == "mmpd":
        print(
            "MMPD Dataset Setup (cross-domain illumination & motion):\n"
            "1. Request access via rPPG-Toolbox / MMPD repo: https://github.com/ubicomplab/rPPG-Toolbox\n"
            f"2. Place .mat files under {out_dir}/MMPD/"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="ubfc", choices=["ubfc", "pure", "ubfc_phys", "mmpd", "mcd", "all"]
    )
    parser.add_argument("--source", type=str, default="hf", choices=["hf", "gdrive"])
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    if args.dataset in ("ubfc", "all"):
        if args.source == "hf":
            download_and_preprocess_ubfc_hf(
                args.out_dir, max_frames=args.max_frames, size=args.size
            )
        else:
            download_ubfc(os.path.join(args.out_dir, "UBFC-rPPG", "1"))
    if args.dataset in ("mmpd", "all") and args.source == "hf":
        download_and_preprocess_mmpd_hf(args.out_dir, max_frames=args.max_frames, size=args.size)
    if args.dataset in ("mcd", "all") and args.source == "hf":
        download_and_preprocess_mcd_hf(args.out_dir, max_frames=args.max_frames, size=args.size)
    for ds in ("pure", "ubfc_phys"):
        if args.dataset in (ds, "all"):
            print()
            print_dataset_instructions(ds, args.out_dir)


if __name__ == "__main__":
    main()
