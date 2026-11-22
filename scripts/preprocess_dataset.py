"""Unified preprocessing: decode videos (UBFC / PURE), extract stabilized face crops & ROI masks."""

import argparse
import glob
import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from roi_extractor import StabilizedFaceROIExtractor, compute_motion_strata  # noqa: E402


def decode_video_to_rgb(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def convert_pure_subject(raw_subj_dir: str, out_subj_dir: str) -> bool:
    """Convert PURE PNG sequence + JSON BVP into unified frames.npy + ground_truth.txt."""
    json_files = glob.glob(os.path.join(raw_subj_dir, "*.json"))
    png_dirs = [d for d in glob.glob(os.path.join(raw_subj_dir, "*")) if os.path.isdir(d)]
    if not json_files or not png_dirs:
        return False

    with open(json_files[0], "r", encoding="utf-8") as f:
        meta = json.load(f)

    pngs = sorted(glob.glob(os.path.join(png_dirs[0], "*.png")))
    if not pngs:
        return False

    frames = np.asarray([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in pngs], dtype=np.uint8)
    full_pack = meta.get("/FullPackage", [])
    bvp_raw = np.array([item["Value"]["waveform"] for item in full_pack], dtype=np.float32)
    bvp_ts = np.array([item["Timestamp"] for item in full_pack], dtype=np.float64)
    img_list = meta.get("/Image", [])
    img_ts = np.array([item["Timestamp"] for item in img_list], dtype=np.float64)

    if len(bvp_raw) >= 2 and len(img_ts) == len(frames):
        bvp_resampled = np.interp(img_ts, bvp_ts, bvp_raw).astype(np.float32)
    else:
        bvp_resampled = np.interp(
            np.linspace(0, 1, len(frames)), np.linspace(0, 1, len(bvp_raw)), bvp_raw
        ).astype(np.float32)

    os.makedirs(out_subj_dir, exist_ok=True)
    np.save(os.path.join(out_subj_dir, "frames.npy"), frames)
    np.savetxt(os.path.join(out_subj_dir, "ground_truth.txt"), bvp_resampled[None, :])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ubfc", choices=["ubfc", "pure"])
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_root = args.out_dir if args.out_dir else args.data_root
    if args.dataset == "pure" and args.out_dir:
        for subj in sorted(os.listdir(args.data_root)):
            raw_dir = os.path.join(args.data_root, subj)
            if os.path.isdir(raw_dir):
                convert_pure_subject(raw_dir, os.path.join(target_root, subj))

    extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
    subjects = sorted(
        s for s in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, s))
    )

    for subj in tqdm(subjects, desc="Preprocessing subjects"):
        subj_dir = os.path.join(target_root, subj)
        frames_path = os.path.join(subj_dir, "frames.npy")
        crop_path = os.path.join(subj_dir, "face_crops.npy")
        mask_path = os.path.join(subj_dir, "region_masks.npy")
        skin_path = os.path.join(subj_dir, "skin_masks.npy")

        if not os.path.exists(frames_path):
            for vname in ("vid.avi", "vid.mp4", "video.avi", "video.mp4"):
                vp = os.path.join(subj_dir, vname)
                if os.path.exists(vp):
                    frames = decode_video_to_rgb(vp)
                    np.save(frames_path, frames)
                    break

        if not os.path.exists(frames_path):
            continue

        if (
            not args.overwrite
            and os.path.exists(crop_path)
            and os.path.exists(mask_path)
            and os.path.exists(skin_path)
        ):
            continue

        frames = np.load(frames_path, mmap_mode="r")
        ext = extractor.extract_sequence_crops(frames, out_size=args.size, detect_stride=2)
        np.save(crop_path, ext["crops"].astype(np.float16))
        np.save(mask_path, (ext["region_masks"] > 0.5).astype(np.uint8))
        np.save(skin_path, (ext["skin_masks"] > 0.5).astype(np.uint8))

        motion = compute_motion_strata(ext["crops"], ext["region_masks"])
        with open(os.path.join(subj_dir, "motion_stats.json"), "w", encoding="utf-8") as f:
            json.dump(motion, f, indent=2)


if __name__ == "__main__":
    main()
