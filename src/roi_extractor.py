"""Stabilized face ROI detection, anatomical region parcellation, H1/H2 mask variants, and motion stratification."""

import os
import urllib.request
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP_SOLUTIONS = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    _HAS_MP_TASKS = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")
    _HAS_MEDIAPIPE = _HAS_MP_SOLUTIONS or _HAS_MP_TASKS
except Exception:
    mp = None
    _HAS_MP_SOLUTIONS = False
    _HAS_MP_TASKS = False
    _HAS_MEDIAPIPE = False


EXPERIMENT_A_VARIANTS = [
    "full_frame_raw",
    "full_frame_static_70",
    "full_frame_bg_removed",
    "face_crop_only",
    "skin_mask_only",
    "face_static_70",
    "mouth_jaw_suppression",
    "forehead_cheeks_only",
    "random_equal_area_mask",
]

REGION_NAMES = ["forehead", "left_cheek", "right_cheek", "mouth_jaw"]

# MediaPipe 468 FaceMesh indices for anatomical regions
_MP_FOREHEAD = [10, 338, 297, 332, 284, 251, 301, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70, 21, 54, 103, 67, 109]
_MP_LEFT_CHEEK = [116, 117, 118, 119, 100, 126, 209, 49, 203, 205, 50, 123, 147, 213, 192, 214, 137]
_MP_RIGHT_CHEEK = [345, 346, 347, 348, 329, 355, 429, 279, 423, 425, 280, 352, 376, 433, 416, 434, 366]
_MP_MOUTH_JAW = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    max_val = float(frame.max()) if frame.size else 1.0
    if max_val <= 1.5:
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)


def compute_skin_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Compute binary skin mask in RGB using YCrCb + HSV chrominance bounds.
    Returns float32 array of shape (H, W) with values in {0.0, 1.0}.
    """
    img_u8 = _to_uint8_rgb(frame_rgb)
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)

    # Standard YCrCb skin thresholds
    lower_ycrcb = np.array([20, 132, 76], dtype=np.uint8)
    upper_ycrcb = np.array([255, 175, 130], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # HSV sanity check against non-skin green/blue walls
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask_hsv = (((h <= 25) | (h >= 160)) & (s >= 20) & (v >= 35)).astype(np.uint8) * 255

    skin = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    return (skin > 0).astype(np.float32)


def make_geometric_region_masks(H: int, W: int) -> np.ndarray:
    """
    Build stabilized anatomical ROI masks (K=4, H, W) inside an aligned face crop:
      0: Forehead
      1: Left Cheek
      2: Right Cheek
      3: Mouth & Jaw
    """
    masks = np.zeros((4, H, W), dtype=np.float32)

    def _fill_ellipse_box(idx: int, y0: float, y1: float, x0: float, x1: float) -> None:
        cy = int(round(0.5 * (y0 + y1) * H))
        cx = int(round(0.5 * (x0 + x1) * W))
        ry = max(2, int(round(0.5 * (y1 - y0) * H)))
        rx = max(2, int(round(0.5 * (x1 - x0) * W)))
        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        masks[idx] = canvas.astype(np.float32)

    # Forehead: upper-central brow to hairline
    _fill_ellipse_box(0, 0.06, 0.30, 0.22, 0.78)
    # Left cheek (image-left)
    _fill_ellipse_box(1, 0.42, 0.67, 0.14, 0.41)
    # Right cheek (image-right)
    _fill_ellipse_box(2, 0.42, 0.67, 0.59, 0.86)
    # Mouth & Jaw
    _fill_ellipse_box(3, 0.66, 0.94, 0.22, 0.78)

    return masks


class StabilizedFaceROIExtractor:
    """
    Temporally stabilized face bounding-box detector and anatomical parcellation.

    Supports MediaPipe FaceMesh when installed, with a reliable OpenCV Haar +
    skin-region fallback smoothed by Exponential Moving Average (EMA) to eliminate
    frame-by-frame bounding-box jitter.
    """

    def __init__(
        self,
        ema_alpha: float = 0.85,
        use_mediapipe: bool = True,
        min_bbox_frac: float = 0.12,
    ):
        self.ema_alpha = ema_alpha
        self.min_bbox_frac = min_bbox_frac
        self.use_mediapipe = bool(use_mediapipe and _HAS_MEDIAPIPE)

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._mp_mesh = None
        self._mp_landmarker = None
        self._yunet = None
        self._facemark = None

        if self.use_mediapipe and _HAS_MP_SOLUTIONS:
            try:
                self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._mp_mesh = None
        elif self.use_mediapipe and _HAS_MP_TASKS:
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                model_dir = os.path.join(repo_root, "models")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "face_landmarker.task")
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                        model_path,
                    )
                base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                )
                self._mp_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            except Exception:
                self._mp_landmarker = None

        if self._mp_mesh is None and self._mp_landmarker is None:
            self.use_mediapipe = False
            if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "face"):
                try:
                    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    model_dir = os.path.join(repo_root, "models")
                    os.makedirs(model_dir, exist_ok=True)
                    yunet_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
                    lbf_path = os.path.join(model_dir, "lbfmodel.yaml")
                    if not os.path.exists(yunet_path):
                        urllib.request.urlretrieve(
                            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                            yunet_path,
                        )
                    if not os.path.exists(lbf_path):
                        urllib.request.urlretrieve(
                            "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml",
                            lbf_path,
                        )
                    self._yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.5, 0.3, 5000)
                    self._facemark = cv2.face.createFacemarkLBF()
                    self._facemark.loadModel(lbf_path)
                except Exception:
                    self._yunet = None
                    self._facemark = None

    def _detect_raw_bbox_and_landmarks(
        self, frame_u8: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        H, W = frame_u8.shape[:2]

        if self._mp_mesh is not None:
            res = self._mp_mesh.process(frame_u8)
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                bw, bh = x2 - x1, y2 - y1
                # Expand slightly upward to guarantee full forehead coverage
                bbox = np.array(
                    [
                        max(0.0, x1 - 0.06 * bw),
                        max(0.0, y1 - 0.12 * bh),
                        min(float(W), x2 + 0.06 * bw),
                        min(float(H), y2 + 0.04 * bh),
                    ],
                    dtype=np.float32,
                )
                return bbox, pts

        if self._mp_landmarker is not None:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_u8))
            res = self._mp_landmarker.detect(mp_img)
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                bw, bh = x2 - x1, y2 - y1
                bbox = np.array(
                    [
                        max(0.0, x1 - 0.06 * bw),
                        max(0.0, y1 - 0.12 * bh),
                        min(float(W), x2 + 0.06 * bw),
                        min(float(H), y2 + 0.04 * bh),
                    ],
                    dtype=np.float32,
                )
                return bbox, pts

        if self._yunet is not None:
            self._yunet.setInputSize((W, H))
            _, faces_yn = self._yunet.detect(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
            if faces_yn is not None and len(faces_yn) > 0:
                best = max(faces_yn, key=lambda b: float(b[2]) * float(b[3]))
                fx, fy, fw, fh = float(best[0]), float(best[1]), float(best[2]), float(best[3])
                pts = None
                if self._facemark is not None:
                    gray_yn = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)
                    box_arr = np.array([[int(round(fx)), int(round(fy)), int(round(fw)), int(round(fh))]], dtype=np.int32)
                    ok, landmarks = self._facemark.fit(gray_yn, box_arr)
                    if ok and len(landmarks) > 0:
                        pts = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 2)
                bbox = np.array(
                    [
                        max(0.0, fx - 0.06 * fw),
                        max(0.0, fy - 0.18 * fh),
                        min(float(W), fx + 1.06 * fw),
                        min(float(H), fy + 1.04 * fh),
                    ],
                    dtype=np.float32,
                )
                return bbox, pts

        gray = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)
        min_side = max(24, int(min(H, W) * self.min_bbox_frac))
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side)
        )
        if len(faces) > 0:
            # Pick largest detected face
            fx, fy, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
            bbox = np.array(
                [
                    max(0.0, fx - 0.05 * fw),
                    max(0.0, fy - 0.12 * fh),
                    min(float(W), fx + 1.05 * fw),
                    min(float(H), fy + 1.04 * fh),
                ],
                dtype=np.float32,
            )
            return bbox, None

        # Fallback: largest connected skin-colored region (robust for stylized/synthetic faces)
        skin = compute_skin_mask(frame_u8)
        if skin.sum() > 0.02 * H * W:
            ys, xs = np.where(skin > 0)
            y1, y2 = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
            x1, x2 = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                return np.array([x1, y1, x2, y2], dtype=np.float32), None

        return None, None

    def extract_sequence_crops(
        self,
        frames: np.ndarray,
        out_size: int = 128,
        detect_stride: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Detect and track a temporally smoothed face bounding box across a frame sequence.

        Returns dict with:
          - crops: (T, out_size, out_size, 3) float32 in [0, 1]
          - bboxes: (T, 4) float32 [x1, y1, x2, y2] in original frame coordinates
          - region_masks: (T, 4, out_size, out_size) float32
          - skin_masks: (T, out_size, out_size) float32
        """
        T, H, W, _ = frames.shape
        fallback_bbox = np.array([W * 0.25, H * 0.12, W * 0.75, H * 0.85], dtype=np.float32)

        raw_bboxes: List[Optional[np.ndarray]] = []
        raw_lms: List[Optional[np.ndarray]] = []
        last_bbox = None
        last_lms = None

        for t in range(T):
            if t % max(1, detect_stride) == 0 or last_bbox is None:
                u8 = _to_uint8_rgb(frames[t])
                bbox, lms = self._detect_raw_bbox_and_landmarks(u8)
                if bbox is not None:
                    last_bbox = bbox
                    last_lms = lms
            raw_bboxes.append(last_bbox)
            raw_lms.append(last_lms)

        # Initialize smoothed bbox with median of valid detections to avoid initial transient jump
        valid_bboxes = [b for b in raw_bboxes if b is not None]
        if valid_bboxes:
            ema_bbox = np.median(np.stack(valid_bboxes, axis=0), axis=0).astype(np.float32)
        else:
            ema_bbox = fallback_bbox.copy()

        # Make square-ish stable aspect ratio & smooth via EMA
        bboxes = np.zeros((T, 4), dtype=np.float32)
        for t in range(T):
            cur = raw_bboxes[t]
            if cur is not None:
                # Reject impossible >35% frame jumps unless consistent
                cur_w = cur[2] - cur[0]
                ema_w = ema_bbox[2] - ema_bbox[0]
                if cur_w > 0 and 0.6 * ema_w <= cur_w <= 1.6 * ema_w:
                    ema_bbox = self.ema_alpha * ema_bbox + (1.0 - self.ema_alpha) * cur
            bboxes[t] = ema_bbox

        # Smooth detected facial landmarks via EMA so anatomical polygons stay jitter-free
        smoothed_lms: List[Optional[np.ndarray]] = [None] * T
        valid_lms = [l for l in raw_lms if l is not None]
        if valid_lms and all(l.shape == valid_lms[0].shape for l in valid_lms):
            ema_lms = np.median(np.stack(valid_lms, axis=0), axis=0).astype(np.float32)
            for t in range(T):
                cur_l = raw_lms[t]
                if cur_l is not None and cur_l.shape == ema_lms.shape:
                    ema_lms = self.ema_alpha * ema_lms + (1.0 - self.ema_alpha) * cur_l
                smoothed_lms[t] = ema_lms.copy()
        else:
            smoothed_lms = raw_lms

        crops = np.zeros((T, out_size, out_size, 3), dtype=np.float32)
        region_masks = np.zeros((T, 4, out_size, out_size), dtype=np.float32)
        skin_masks = np.zeros((T, out_size, out_size), dtype=np.float32)
        geom_template = make_geometric_region_masks(out_size, out_size)

        for t in range(T):
            x1, y1, x2, y2 = bboxes[t]
            ix1 = max(0, int(np.floor(x1)))
            iy1 = max(0, int(np.floor(y1)))
            ix2 = min(W, int(np.ceil(x2)))
            iy2 = min(H, int(np.ceil(y2)))
            if ix2 - ix1 < 16 or iy2 - iy1 < 16:
                ix1, iy1, ix2, iy2 = int(W * 0.2), int(H * 0.1), int(W * 0.8), int(H * 0.9)

            crop = frames[t, iy1:iy2, ix1:ix2, :]
            if crop.dtype == np.uint8 or crop.max() > 1.5:
                crop_f = crop.astype(np.float32) / 255.0
            else:
                crop_f = crop.astype(np.float32)

            crop_resized = cv2.resize(crop_f, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            crops[t] = np.clip(crop_resized, 0.0, 1.0)
            skin_masks[t] = compute_skin_mask(crops[t])

            lms = smoothed_lms[t]
            if lms is not None and lms.shape[0] >= 468:
                bw = max(1.0, float(ix2 - ix1))
                bh = max(1.0, float(iy2 - iy1))
                rm = np.zeros((4, out_size, out_size), dtype=np.float32)
                for r_idx, idx_list in enumerate(
                    [_MP_FOREHEAD, _MP_LEFT_CHEEK, _MP_RIGHT_CHEEK, _MP_MOUTH_JAW]
                ):
                    poly = np.stack(
                        [
                            np.clip((lms[idx_list, 0] - ix1) / bw * out_size, 0, out_size - 1),
                            np.clip((lms[idx_list, 1] - iy1) / bh * out_size, 0, out_size - 1),
                        ],
                        axis=1,
                    ).astype(np.int32)
                    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
                    hull = cv2.convexHull(poly)
                    cv2.fillPoly(canvas, [hull], 1)
                    rm[r_idx] = canvas.astype(np.float32) if canvas.sum() > 0 else geom_template[r_idx]
                region_masks[t] = rm
            elif lms is not None and lms.shape[0] == 68:
                bw = max(1.0, float(ix2 - ix1))
                bh = max(1.0, float(iy2 - iy1))
                pts_crop = np.stack(
                    [
                        np.clip((lms[:, 0] - ix1) / bw * out_size, 0.0, float(out_size - 1)),
                        np.clip((lms[:, 1] - iy1) / bh * out_size, 0.0, float(out_size - 1)),
                    ],
                    axis=1,
                )
                brows = pts_crop[17:27]
                face_h = max(12.0, float(pts_crop[8, 1] - brows[:, 1].mean()))
                brows_top = np.stack(
                    [brows[:, 0], np.clip(brows[:, 1] - 0.35 * face_h, 2.0, float(out_size - 2))],
                    axis=1,
                )
                poly_list = [
                    np.concatenate([brows, brows_top], axis=0),
                    pts_crop[[1, 2, 3, 4, 31, 48, 36, 41]],
                    pts_crop[[15, 14, 13, 12, 35, 54, 45, 46]],
                    pts_crop[[5, 6, 7, 8, 9, 10, 11, 48, 49, 50, 51, 52, 53, 54]],
                ]
                rm = np.zeros((4, out_size, out_size), dtype=np.float32)
                for r_idx, p_arr in enumerate(poly_list):
                    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
                    hull = cv2.convexHull(np.rint(p_arr).astype(np.int32))
                    cv2.fillPoly(canvas, [hull], 1)
                    rm[r_idx] = canvas.astype(np.float32) if canvas.sum() > 0 else geom_template[r_idx]
                region_masks[t] = rm
            else:
                region_masks[t] = geom_template

        return {
            "crops": crops,
            "bboxes": bboxes,
            "region_masks": region_masks,
            "skin_masks": skin_masks,
        }


def build_variant_masks(
    variant: str,
    full_frames: Optional[np.ndarray],
    face_crops: np.ndarray,
    region_masks: np.ndarray,
    skin_masks: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Construct the (frames_to_use, spatial_masks) pair for each Experiment A variant.

    Returns compact 2D (H, W) or 3D (T, H, W) masks (or None for unmasked full average)
    without duplicating full video arrays in memory.
    """
    if variant not in EXPERIMENT_A_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {EXPERIMENT_A_VARIANTS}")

    if variant in ("full_frame_raw", "full_frame_static_70", "full_frame_bg_removed"):
        if full_frames is None:
            raise ValueError(f"full_frames required for variant '{variant}'")
        T, H, W, _ = full_frames.shape
        if variant == "full_frame_raw":
            return full_frames, None
        if variant == "full_frame_bg_removed":
            # Skin-select directly on uncropped full frames (isolates background removal without cropping/resampling)
            bg_mask = np.empty((T, H, W), dtype=np.float32)
            for t_idx in range(T):
                m_t = compute_skin_mask(full_frames[t_idx])
                if float(m_t.sum()) < 10.0:
                    m_t = np.ones((H, W), dtype=np.float32)
                bg_mask[t_idx] = m_t
            return full_frames, bg_mask
        m2 = np.ones((H, W), dtype=np.float32)
        m2[int(H * 0.7):, :] = 0.0
        return full_frames, m2

    T, H, W, _ = face_crops.shape
    vid = face_crops

    if variant == "face_crop_only":
        return vid, None
    if variant == "skin_mask_only":
        m = np.asarray(skin_masks, dtype=np.float32).copy()
        empty = m.sum(axis=(1, 2)) < 10
        m[empty] = 1.0
        return vid, m
    if variant == "face_static_70":
        m2 = np.ones((H, W), dtype=np.float32)
        m2[int(H * 0.7):, :] = 0.0
        return vid, m2
    if variant == "mouth_jaw_suppression":
        # Suppress anatomical mouth & jaw region (R3) directly (H1 test)
        m = np.clip(1.0 - np.asarray(region_masks[:, 3], dtype=np.float32), 0.0, 1.0)
        return vid, m
    if variant == "forehead_cheeks_only":
        # Retain only physiologically stable upper-face ROIs (R0 + R1 + R2)
        rm = np.asarray(region_masks, dtype=np.float32)
        m = np.clip(rm[:, 0] + rm[:, 1] + rm[:, 2], 0.0, 1.0)
        return vid, m
    if variant == "random_equal_area_mask":
        # Suppress random spatially static pixels matching exact retained area of mouth_jaw_suppression
        mouth_area_frac = float(np.asarray(region_masks[:, 3], dtype=np.float32).mean())
        keep_prob = float(np.clip(1.0 - mouth_area_frac, 0.2, 0.95))
        rng = np.random.default_rng(seed)
        m2 = (rng.random((H, W)) < keep_prob).astype(np.float32)
        return vid, m2

    raise ValueError(variant)


def compute_motion_strata(
    face_crops: np.ndarray,
    region_masks: np.ndarray,
) -> Dict[str, float]:
    """
    Measure localized mouth/jaw motion vs upper-face global motion using dense optical flow.

    Returns scalar summary metrics per clip/sequence for motion-stratified robustness analysis.
    """
    T, H, W, _ = face_crops.shape
    if T < 2:
        return {"mouth_motion": 0.0, "upper_motion": 0.0, "mouth_to_upper_ratio": 1.0, "total_motion": 0.0}

    mouth_energies = []
    upper_energies = []
    total_energies = []

    prev_gray = cv2.cvtColor(_to_uint8_rgb(face_crops[0]), cv2.COLOR_RGB2GRAY)
    step = max(1, T // 150)  # Fast uniform sampling for long sequences
    for t in range(step, T, step):
        curr_gray = cv2.cvtColor(_to_uint8_rgb(face_crops[t]), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 2, 11, 2, 5, 1.1, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) / float(step)

        m_mask = region_masks[t, 3] > 0.5
        u_mask = (region_masks[t, 0] + region_masks[t, 1] + region_masks[t, 2]) > 0.5

        mouth_energies.append(float(mag[m_mask].mean()) if m_mask.any() else float(mag.mean()))
        upper_energies.append(float(mag[u_mask].mean()) if u_mask.any() else float(mag.mean()))
        total_energies.append(float(mag.mean()))
        prev_gray = curr_gray

    m_mean = float(np.mean(mouth_energies))
    u_mean = float(np.mean(upper_energies))
    return {
        "mouth_motion": m_mean,
        "upper_motion": u_mean,
        "mouth_to_upper_ratio": float(m_mean / (u_mean + 1e-6)),
        "total_motion": float(np.mean(total_energies)),
    }
