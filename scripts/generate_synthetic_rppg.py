"""Generate a physically realistic 42-subject synthetic rPPG dataset for end-to-end verification."""

import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Exact 42 subject IDs from UBFC-rPPG (matching results/static_final_results.csv & SKIA v1 split)
UBFC_42_SUBJECTS = [
    "subject1", "subject3", "subject4", "subject5", "subject8", "subject9",
    "subject10", "subject11", "subject12", "subject13", "subject14", "subject15",
    "subject16", "subject17", "subject18", "subject20", "subject22", "subject23",
    "subject24", "subject25", "subject26", "subject27", "subject30", "subject31",
    "subject32", "subject33", "subject34", "subject35", "subject36", "subject37",
    "subject38", "subject39", "subject40", "subject41", "subject42", "subject43",
    "subject44", "subject45", "subject46", "subject47", "subject48", "subject49",
]


def generate_subject(
    out_dir: str,
    subj_name: str,
    subj_idx: int,
    num_frames: int = 320,
    H: int = 144,
    W: int = 192,
    fs: float = 30.0,
) -> None:
    rng = np.random.default_rng(1000 + subj_idx)
    subj_dir = os.path.join(out_dir, subj_name)
    os.makedirs(subj_dir, exist_ok=True)

    t = np.arange(num_frames, dtype=np.float64) / fs
    hr_bpm = float(rng.uniform(58.0, 122.0))
    f_hr = hr_bpm / 60.0
    phase = float(rng.uniform(0, 2 * np.pi))

    # Ground-truth cardiac BVP signal (fundamental + second harmonic)
    bvp = np.sin(2 * np.pi * f_hr * t + phase) + 0.25 * np.sin(4 * np.pi * f_hr * t + 2 * phase - 0.5)
    bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-8)

    # Half of the subjects exhibit strong speaking / mouth-jaw artifacts inside cardiac band
    is_high_speech = (subj_idx % 2 == 1)
    speech_freq1 = float(rng.uniform(0.85, 1.85))
    speech_freq2 = float(rng.uniform(1.10, 2.15))
    mouth_osc = (
        0.5 + 0.5 * np.sin(2 * np.pi * speech_freq1 * t + 0.3) * np.cos(2 * np.pi * 0.35 * t)
        if is_high_speech
        else 0.1 + 0.08 * np.sin(2 * np.pi * 0.4 * t)
    )
    # Shirt / background sway artifact in lower 30% of full frame (tests H2 confounder)
    shirt_osc = np.sin(2 * np.pi * speech_freq2 * t + 1.1)

    # Base camera frame coordinates
    yy, xx = np.mgrid[0:H, 0:W]
    face_cx, face_cy = W * 0.50, H * 0.43
    face_rx, face_ry = W * 0.17, H * 0.31

    # Skin ellipse & region weights (strongest hemoglobin pulse in forehead & cheeks)
    face_dist = ((xx - face_cx) / face_rx) ** 2 + ((yy - face_cy) / face_ry) ** 2
    face_mask = (face_dist <= 1.0).astype(np.float32)

    # Anatomical sub-regions inside face
    forehead_mask = face_mask * ((yy >= face_cy - 0.90 * face_ry) & (yy <= face_cy - 0.25 * face_ry)).astype(np.float32)
    cheeks_mask = face_mask * ((yy > face_cy - 0.25 * face_ry) & (yy <= face_cy + 0.35 * face_ry)).astype(np.float32)
    mouth_jaw_mask = face_mask * ((yy > face_cy + 0.35 * face_ry) & (yy <= face_cy + 0.98 * face_ry)).astype(np.float32)

    # Shirt / torso occupies bottom 30% of full camera frame (y >= 0.70*H)
    torso_mask = ((yy >= int(H * 0.70)) & (np.abs(xx - face_cx) <= W * 0.26)).astype(np.float32) * (1.0 - face_mask)
    bg_mask = np.clip(1.0 - face_mask - torso_mask, 0.0, 1.0)

    # Physiological pulse amplitude map (strong in forehead/cheeks, weaker/noisier in jaw)
    pulse_gain = (
        1.0 * forehead_mask
        + 0.95 * cheeks_mask
        + 0.35 * mouth_jaw_mask
    )

    # Skin RGB (inside YCrCb/HSV skin range): R=202, G=148, B=124
    skin_base = np.array([202.0, 148.0, 124.0], dtype=np.float32)
    bg_base = np.array([58.0, 152.0, 128.0], dtype=np.float32)     # UBFC-like teal wall
    shirt_base = np.array([75.0, 95.0, 150.0], dtype=np.float32)

    # Hemoglobin pulsatile RGB reflectance vector (green strongest)
    pulse_rgb = np.array([-1.3, -2.8, -0.9], dtype=np.float32)

    frames = np.zeros((num_frames, H, W, 3), dtype=np.uint8)
    for k in range(num_frames):
        # Gentle head micro-sway (1-2 px)
        dx = float(1.2 * np.sin(2 * np.pi * 0.28 * t[k] + phase))
        dy = float(0.8 * np.cos(2 * np.pi * 0.22 * t[k]))

        img = (
            bg_mask[..., None] * bg_base
            + torso_mask[..., None] * (shirt_base + 14.0 * shirt_osc[k] * np.array([0.8, -0.5, 0.4], dtype=np.float32))
            + face_mask[..., None] * skin_base
        )

        # Add clean cardiac BVP modulation over face skin
        img += (pulse_gain[..., None] * bvp[k]) * pulse_rgb

        # Add realistic lower-face speaking motion artifact (lip opening + shadow variation)
        if is_high_speech:
            lip_open_ry = max(1.5, 5.5 * mouth_osc[k])
            mouth_cy = face_cy + 0.56 * face_ry
            lip_dist = ((xx - face_cx) / (face_rx * 0.42)) ** 2 + ((yy - mouth_cy) / lip_open_ry) ** 2
            lip_region = (lip_dist <= 1.0).astype(np.float32)
            # Dark mouth interior + color shift across jaw from non-rigid muscle shadow
            img = img * (1.0 - 0.55 * lip_region[..., None]) + lip_region[..., None] * np.array([90.0, 50.0, 50.0], dtype=np.float32)
            img += (mouth_jaw_mask[..., None] * (12.0 * np.sin(2 * np.pi * speech_freq1 * t[k]))) * np.array([1.0, -0.6, 0.3], dtype=np.float32)
        else:
            # Subtle static closed mouth feature
            mouth_cy = face_cy + 0.56 * face_ry
            lip_dist = ((xx - face_cx) / (face_rx * 0.38)) ** 2 + ((yy - mouth_cy) / 1.8) ** 2
            lip_region = (lip_dist <= 1.0).astype(np.float32)
            img = img * (1.0 - 0.25 * lip_region[..., None])

        # Eyes & brows (helps visual realism and Haar/landmark consistency)
        for ex in (face_cx - 0.36 * face_rx, face_cx + 0.36 * face_rx):
            eye_dist = ((xx - ex) / 3.2) ** 2 + ((yy - (face_cy - 0.14 * face_ry)) / 2.2) ** 2
            eye_mask = (eye_dist <= 1.0).astype(np.float32)
            img = img * (1.0 - 0.65 * eye_mask[..., None])

        # Shift subtly for realism and add mild sensor noise
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        noise = rng.normal(0.0, 0.8, size=(H, W, 3)).astype(np.float32)
        frames[k] = np.clip(img + noise, 0.0, 255.0).astype(np.uint8)

    np.save(os.path.join(subj_dir, "frames.npy"), frames)
    hr_trace = np.full(num_frames, hr_bpm, dtype=np.float32)
    gt = np.vstack([bvp.astype(np.float32), hr_trace, t.astype(np.float32)])
    np.savetxt(os.path.join(subj_dir, "ground_truth.txt"), gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/SYNTH-UBFC")
    parser.add_argument("--num-subjects", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=320)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = UBFC_42_SUBJECTS[: args.num_subjects]
    for i, name in enumerate(tqdm(names, desc="Generating SYNTH-UBFC")):
        generate_subject(args.out_dir, name, i, num_frames=args.num_frames)
    print(f"Generated {len(names)} subjects in {args.out_dir}")


if __name__ == "__main__":
    main()
