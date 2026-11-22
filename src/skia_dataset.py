"""Datasets for SKIA v1 (legacy) and SKIA v2 (stabilized face crops + anatomical ROI masks)."""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.lib.format as npy_fmt
import torch
from torch.utils.data import Dataset

from roi_extractor import StabilizedFaceROIExtractor


def _read_npy_len(path: str) -> int:
    """Read leading dimension length of a .npy file without mapping its payload."""
    with open(path, "rb") as f:
        v = npy_fmt.read_magic(f)
        if v == (1, 0):
            shape, _, _ = npy_fmt.read_array_header_1_0(f)
            return int(shape[0])
        if v == (2, 0):
            shape, _, _ = npy_fmt.read_array_header_2_0(f)
            return int(shape[0])
    arr = np.load(path, mmap_mode="r")
    return int(arr.shape[0])


def _read_npy_slice(path: str, start: int, end: int) -> np.ndarray:
    """Read contiguous slice [start:end] from a C-order .npy file with zero mmap leaks."""
    with open(path, "rb") as f:
        v = npy_fmt.read_magic(f)
        if v == (1, 0):
            shape, fortran, dtype = npy_fmt.read_array_header_1_0(f)
        elif v == (2, 0):
            shape, fortran, dtype = npy_fmt.read_array_header_2_0(f)
        else:
            arr = np.load(path, mmap_mode="r")
            return np.asarray(arr[start:end], dtype=np.float32)

        if fortran:
            arr = np.load(path, mmap_mode="r")
            return np.asarray(arr[start:end], dtype=np.float32)

        offset = f.tell()
        row_elems = int(np.prod(shape[1:]))
        count = int(end - start)
        f.seek(offset + int(start) * row_elems * dtype.itemsize)
        raw = np.fromfile(f, dtype=dtype, count=count * row_elems)
        return raw.reshape((count,) + shape[1:]).astype(np.float32, copy=False)


def load_subject_bvp(subj_dir: str) -> np.ndarray:
    """Load ground-truth 1D BVP waveform from UBFC/PURE/synthetic formats."""
    gt_txt = os.path.join(subj_dir, "ground_truth.txt")
    if os.path.exists(gt_txt):
        gt = np.loadtxt(gt_txt)
        if gt.ndim == 2:
            return gt[0].astype(np.float32)
        return gt.astype(np.float32)

    for fname in ("bvp.npy", "gt.npy"):
        p = os.path.join(subj_dir, fname)
        if os.path.exists(p):
            arr = np.load(p)
            return arr.reshape(-1).astype(np.float32)

    raise FileNotFoundError(f"No ground truth BVP found in {subj_dir}")


class SKIADataset(Dataset):
    """Legacy SKIA v1 dataset (with optional metadata for subject stitching)."""

    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        clip_len: int = 128,
        stride: int = 64,
        size: int = 128,
        mode: str = "raw",
        return_meta: bool = False,
    ):
        self.data_root = data_root
        self.subjects = subjects
        self.clip_len = clip_len
        self.stride = stride
        self.size = size
        self.mode = mode
        self.return_meta = return_meta
        self.samples: List[Tuple[str, int]] = []
        self._bvp_cache: Dict[str, np.ndarray] = {}

        for subj in subjects:
            subj_dir = os.path.join(data_root, subj)
            frames_path = os.path.join(subj_dir, "frames.npy")
            if not os.path.exists(frames_path):
                continue
            try:
                bvp = load_subject_bvp(subj_dir)
            except FileNotFoundError:
                continue

            self._bvp_cache[subj] = bvp
            T = min(_read_npy_len(frames_path), len(bvp))
            for start in range(0, T - clip_len + 1, stride):
                self.samples.append((subj, start))

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_center_crop(self, x: np.ndarray) -> torch.Tensor:
        T, H, W, C = x.shape
        top = int(H * 0.1)
        bottom = int(H * 0.9)
        left = int(W * 0.2)
        right = int(W * 0.8)
        x = x[:, top:bottom, left:right, :]
        t_x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
        t_x = torch.nn.functional.interpolate(
            t_x, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        return t_x

    def _static_mask(self, T: int, H: int, W: int) -> torch.Tensor:
        mask = torch.ones((T, 1, H, W), dtype=torch.float32)
        mask[:, :, int(H * 0.7):, :] = 0.0
        return mask

    def __getitem__(self, idx: int):
        subj, start = self.samples[idx]
        subj_dir = os.path.join(self.data_root, subj)

        clip = _read_npy_slice(os.path.join(subj_dir, "frames.npy"), start, start + self.clip_len)
        if clip.max() > 1.5:
            clip = clip / 255.0
        bvp = self._bvp_cache[subj][start:start + self.clip_len]

        x = self._resize_center_crop(clip)
        x = (x - x.mean()) / (x.std() + 1e-6)

        T, C, H, W = x.shape
        static_mask = self._static_mask(T, H, W)

        raw = x
        masked = x * static_mask
        diff = torch.zeros_like(x)
        diff[1:] = x[1:] - x[:-1]

        if self.mode == "raw":
            inp = raw
        elif self.mode == "masked":
            inp = masked
        elif self.mode == "raw_masked":
            inp = torch.cat([raw, masked], dim=1)
        elif self.mode == "full":
            inp = torch.cat([raw, masked, diff], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        bvp_t = torch.from_numpy(bvp.copy())
        bvp_t = (bvp_t - bvp_t.mean()) / (bvp_t.std() + 1e-6)
        inp = inp.permute(1, 0, 2, 3)  # (C, T, H, W)

        if self.return_meta:
            return inp, bvp_t, subj, int(start)
        return inp, bvp_t


class SKIAV2Dataset(Dataset):
    """
    Standardized dataset for SKIA-v2 and all deep baselines.

    Uses stabilized face crops (`face_crops.npy`) and 4 anatomical ROI masks
    (`region_masks.npy`) read directly via zero-mmap offset slices.
    """

    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        clip_len: int = 128,
        stride: int = 64,
        size: int = 128,
        return_masks: bool = True,
        return_meta: bool = False,
        auto_cache_crops: bool = True,
    ):
        self.data_root = data_root
        self.subjects = subjects
        self.clip_len = clip_len
        self.stride = stride
        self.size = size
        self.return_masks = return_masks
        self.return_meta = return_meta
        self.auto_cache_crops = auto_cache_crops
        self.skip_masks: bool = False
        self.fast_gpu_prep: bool = False
        self._bits_t = torch.arange(8, dtype=torch.uint8).view(8, 1, 1, 1)
        self.samples: List[Tuple[str, int]] = []
        self._bvp_cache: Dict[str, np.ndarray] = {}
        self._mem_cache: Dict[str, Tuple[Any, ...]] = {}
        self._extractor: Optional[StabilizedFaceROIExtractor] = None

        for subj in subjects:
            subj_dir = os.path.join(data_root, subj)
            crop_path = os.path.join(subj_dir, "face_crops.npy")
            mask_path = os.path.join(subj_dir, "region_masks.npy")
            frames_path = os.path.join(subj_dir, "frames.npy")

            if not (os.path.exists(crop_path) and os.path.exists(mask_path)):
                if not os.path.exists(frames_path):
                    continue
                self._ensure_cached(subj_dir)

            try:
                bvp = load_subject_bvp(subj_dir)
            except FileNotFoundError:
                continue

            self._bvp_cache[subj] = bvp
            T = min(_read_npy_len(crop_path), len(bvp))
            if T <= 1200:
                crops_arr = np.load(crop_path)
                if crops_arr.dtype != np.uint8:
                    scale = 255.0 if float(crops_arr.max()) <= 1.5 else 1.0
                    crops_u8 = np.clip(np.rint(crops_arr * scale), 0, 255).astype(np.uint8)
                else:
                    crops_u8 = crops_arr
                if crops_u8.ndim == 4 and crops_u8.shape[-1] == 3:
                    crops_u8 = np.ascontiguousarray(crops_u8.transpose(3, 0, 1, 2))
                masks_arr = np.load(mask_path)
                num_reg = int(masks_arr.shape[1])
                if num_reg <= 8:
                    packed = np.zeros(
                        (len(masks_arr), masks_arr.shape[2], masks_arr.shape[3]), dtype=np.uint8
                    )
                    for k in range(num_reg):
                        packed[masks_arr[:, k] > 0.5] |= np.uint8(1 << k)
                    self._mem_cache[subj] = (crops_u8, packed, num_reg)
                else:
                    masks_u8 = (masks_arr > 0.5).astype(np.uint8) if masks_arr.dtype != np.uint8 else masks_arr
                    self._mem_cache[subj] = (crops_u8, masks_u8, 0)
                del crops_arr, masks_arr

            for start in range(0, T - clip_len + 1, stride):
                self.samples.append((subj, start))

    def __len__(self) -> int:
        return len(self.samples)

    def _ensure_cached(self, subj_dir: str) -> None:
        if self._extractor is None:
            self._extractor = StabilizedFaceROIExtractor(ema_alpha=0.85, use_mediapipe=True)
        frames = np.load(os.path.join(subj_dir, "frames.npy"), mmap_mode="r")
        extracted = self._extractor.extract_sequence_crops(frames, out_size=self.size, detect_stride=2)
        np.save(os.path.join(subj_dir, "face_crops.npy"), extracted["crops"].astype(np.float16))
        np.save(os.path.join(subj_dir, "region_masks.npy"), (extracted["region_masks"] > 0.5).astype(np.uint8))
        np.save(os.path.join(subj_dir, "skin_masks.npy"), (extracted["skin_masks"] > 0.5).astype(np.uint8))

    def __getitem__(self, idx: int):
        subj, start = self.samples[idx]
        subj_dir = os.path.join(self.data_root, subj)

        if subj in self._mem_cache:
            crops_u8, m_cached, num_reg = self._mem_cache[subj]
            if self.fast_gpu_prep and self.size == crops_u8.shape[-1] and (self.skip_masks or not self.return_masks or num_reg == 4):
                x = torch.from_numpy(crops_u8[:, start:start + self.clip_len])  # (3, T, H, W) uint8
                if self.return_masks and not self.skip_masks:
                    m = torch.from_numpy(m_cached[start:start + self.clip_len])  # (T, H, W) bit-packed uint8
                else:
                    m = torch.empty(0, dtype=torch.uint8)
                bvp = self._bvp_cache[subj][start:start + self.clip_len]
                bvp_t = torch.from_numpy(bvp.copy())
                bvp_t = (bvp_t - bvp_t.mean()) / (bvp_t.std() + 1e-6)
                if self.return_masks and self.return_meta:
                    return x, m, bvp_t, subj, int(start)
                if self.return_masks:
                    return x, m, bvp_t
                if self.return_meta:
                    return x, bvp_t, subj, int(start)
                return x, bvp_t

            x = torch.from_numpy(crops_u8[:, start:start + self.clip_len]).float() / 255.0  # (3, T, H, W)
            if self.return_masks and not self.skip_masks:
                if num_reg > 0:
                    p_slice = torch.from_numpy(m_cached[start:start + self.clip_len]).unsqueeze(0)
                    m = ((p_slice >> self._bits_t[:num_reg]) & 1).float()  # (K, T, H, W)
                else:
                    m = torch.from_numpy(m_cached[start:start + self.clip_len]).permute(1, 0, 2, 3).float()
            else:
                m = torch.empty(0, dtype=torch.float32)
        else:
            clip = _read_npy_slice(os.path.join(subj_dir, "face_crops.npy"), start, start + self.clip_len)
            if clip.size and float(clip.max()) > 1.5:
                clip = clip / 255.0
            x = torch.from_numpy(np.ascontiguousarray(clip)).permute(3, 0, 1, 2).float()  # (3, T, H, W)
            if self.return_masks and not self.skip_masks:
                rmask = _read_npy_slice(os.path.join(subj_dir, "region_masks.npy"), start, start + self.clip_len)
                m = torch.from_numpy(np.ascontiguousarray(rmask)).permute(1, 0, 2, 3).float()  # (K, T, H, W)
            else:
                m = torch.empty(0, dtype=torch.float32)
        bvp = self._bvp_cache[subj][start:start + self.clip_len]

        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = torch.nn.functional.interpolate(
                x.permute(1, 0, 2, 3), size=(self.size, self.size), mode="bilinear", align_corners=False
            ).permute(1, 0, 2, 3)
            if m.numel() > 0:
                m = torch.nn.functional.interpolate(
                    m.permute(1, 0, 2, 3), size=(self.size, self.size), mode="nearest"
                ).permute(1, 0, 2, 3)

        x = (x - x.mean()) / (x.std() + 1e-6)

        bvp_t = torch.from_numpy(bvp.copy())
        bvp_t = (bvp_t - bvp_t.mean()) / (bvp_t.std() + 1e-6)

        if self.return_masks and self.return_meta:
            return x, m, bvp_t, subj, int(start)
        if self.return_masks:
            return x, m, bvp_t
        if self.return_meta:
            return x, bvp_t, subj, int(start)
        return x, bvp_t
