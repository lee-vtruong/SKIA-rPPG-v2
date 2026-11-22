"""Subject-independent dataset splitting utilities (leak-free Train/Val/Test and K-fold)."""

import os
from typing import Dict, List, Tuple
import numpy as np


UBFC_V1_TEST_SUBJECTS = [
    "subject43",
    "subject44",
    "subject45",
    "subject46",
    "subject47",
    "subject48",
    "subject49",
    "subject5",
    "subject8",
    "subject9",
]


def discover_subjects(data_root: str) -> List[str]:
    """Discover valid subject directories sorted alphabetically."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    subjects = []
    for name in sorted(os.listdir(data_root)):
        subj_dir = os.path.join(data_root, name)
        if not os.path.isdir(subj_dir):
            continue
        has_frames = os.path.exists(os.path.join(subj_dir, "frames.npy"))
        has_video = any(
            os.path.exists(os.path.join(subj_dir, v))
            for v in ("vid.avi", "vid.mp4", "video.avi", "video.mp4")
        )
        has_gt = any(
            os.path.exists(os.path.join(subj_dir, g))
            for g in ("ground_truth.txt", "bvp.npy", "gt.npy")
        )
        if (has_frames or has_video) and has_gt:
            subjects.append(name)
    return subjects


def get_subject_split(
    subjects: List[str],
    seed: int = 42,
    val_ratio_of_trainval: float = 0.1875,
    test_ratio: float = 0.24,
    preserve_v1_test: bool = True,
) -> Dict[str, List[str]]:
    """
    Create a strict subject-independent Train / Val / Test split.

    When `preserve_v1_test=True` and the standard UBFC-rPPG test subjects
    (or >= 42 subjects) are present, the exact 10 test subjects from SKIA v1
    are held out as the fixed test split, and the remaining 32 subjects are
    split deterministically into 26 Train + 6 Validation subjects.
    """
    subjects = sorted(subjects)
    n_total = len(subjects)
    if n_total < 3:
        raise ValueError(f"Need at least 3 subjects for train/val/test, got {n_total}")

    if preserve_v1_test and all(s in subjects for s in UBFC_V1_TEST_SUBJECTS):
        test_subjs = [s for s in subjects if s in UBFC_V1_TEST_SUBJECTS]
        trainval_subjs = [s for s in subjects if s not in UBFC_V1_TEST_SUBJECTS]
    else:
        n_test = max(1, int(round(n_total * test_ratio)))
        if n_total - n_test < 2:
            n_test = 1
        if preserve_v1_test and n_total >= 10:
            # Alphabetical tail holdout matching SKIA v1 convention
            test_subjs = subjects[-n_test:]
            trainval_subjs = subjects[:-n_test]
        else:
            rng = np.random.default_rng(seed)
            perm = list(rng.permutation(subjects))
            test_subjs = sorted(perm[:n_test])
            trainval_subjs = sorted(perm[n_test:])

    rng = np.random.default_rng(seed)
    perm_tv = list(rng.permutation(trainval_subjs))
    n_val = max(1, int(round(len(trainval_subjs) * val_ratio_of_trainval)))
    if len(trainval_subjs) - n_val < 1:
        n_val = 1

    val_subjs = sorted(perm_tv[:n_val])
    train_subjs = sorted(perm_tv[n_val:])

    # Sanity verification: zero overlap
    assert set(train_subjs).isdisjoint(val_subjs)
    assert set(train_subjs).isdisjoint(test_subjs)
    assert set(val_subjs).isdisjoint(test_subjs)

    return {
        "train": train_subjs,
        "val": val_subjs,
        "test": test_subjs,
    }


def get_kfold_splits(
    subjects: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """Generate subject-independent K-fold Train/Val/Test splits."""
    subjects = sorted(subjects)
    if len(subjects) < n_splits:
        raise ValueError(f"Cannot create {n_splits} folds from {len(subjects)} subjects")

    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(subjects))
    folds = [sorted(list(arr)) for arr in np.array_split(perm, n_splits)]

    splits = []
    for test_idx in range(n_splits):
        val_idx = (test_idx + 1) % n_splits
        test_subjs = folds[test_idx]
        val_subjs = folds[val_idx]
        train_subjs = sorted(
            s for i, fold in enumerate(folds) if i not in (test_idx, val_idx) for s in fold
        )
        splits.append({"train": train_subjs, "val": val_subjs, "test": test_subjs})
    return splits
