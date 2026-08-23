"""
NeoGuard - Color & Texture Feature Extraction Module
--------------------------------------------------------
Implements the "Color and Texture Feature Extraction" stage (deck page 6, step 4).

Clinical grounding: bilirubin deposition shifts tissue color toward yellow. In the
CIE Lab color space, the b* channel directly encodes the blue-yellow axis, so b* (and
related yellowness indices used in colorimetry/dermatology literature, e.g. ASTM
E313-style whiteness/yellowness indices) are natural, physically-motivated features
-- rather than asking a model to learn color relationships from raw RGB alone. This
mirrors how prior smartphone jaundice-screening research (e.g. BiliCam, University of
Washington) extracted explicit color-space features before regression rather than
running an end-to-end black-box model on raw pixels.

This module extracts, from the calibrated ROI:
  - Mean/std color in RGB, HSV, and Lab spaces
  - A yellowness index derived from Lab b*/L*
  - Simple texture statistics (local contrast) as a secondary signal for image
    quality / skin texture effects
"""

import cv2
import numpy as np


def _masked_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an (N, C) array of pixels where mask is nonzero."""
    return image[mask > 0]


def extract_features(bgr_image: np.ndarray, roi_mask: np.ndarray) -> dict:
    """
    Extract a feature vector (as a dict) from the calibrated ROI.

    Parameters
    ----------
    bgr_image : np.ndarray
        Calibrated (color-corrected) capture, BGR uint8.
    roi_mask : np.ndarray
        Binary mask (0/255) marking the sclera or skin region.

    Returns
    -------
    dict of feature_name -> float
    """
    if np.count_nonzero(roi_mask) == 0:
        raise ValueError("empty ROI mask - cannot extract features")

    bgr_px = _masked_pixels(bgr_image, roi_mask).astype(np.float32)

    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    lab_px = _masked_pixels(lab, roi_mask).astype(np.float32)
    # OpenCV Lab: L in [0,255] (scale of 0-100), a,b in [0,255] with 128 offset
    L = lab_px[:, 0] * (100.0 / 255.0)
    a = lab_px[:, 1] - 128.0
    b = lab_px[:, 2] - 128.0

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_px = _masked_pixels(hsv, roi_mask).astype(np.float32)

    B, G, R = bgr_px[:, 0], bgr_px[:, 1], bgr_px[:, 2]

    # Yellowness index: positive b* = yellow, negative = blue. Normalize by L* so
    # brightness differences (exposure residuals) don't masquerade as yellowness.
    yellowness_index = float(np.mean(b / np.clip(L, 1e-3, None)) * 100.0)

    # Secondary classic colorimetry-style whiteness/yellowness approximation
    # (loosely modeled on ASTM E313 form: WI ~ related to L, a, b)
    astm_yellowness = float(np.mean(1.28 * (lab_px[:, 2] - 128.0) - 1.06 * (lab_px[:, 1] - 128.0)))

    # Texture: local contrast (std of grayscale) within ROI as a simple proxy;
    # a real system would add GLCM contrast/homogeneity/entropy.
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_px = _masked_pixels(gray, roi_mask).astype(np.float32)
    texture_std = float(np.std(gray_px))

    features = {
        "mean_R": float(np.mean(R)), "mean_G": float(np.mean(G)), "mean_B": float(np.mean(B)),
        "std_R": float(np.std(R)), "std_G": float(np.std(G)), "std_B": float(np.std(B)),
        "mean_H": float(np.mean(hsv_px[:, 0])), "mean_S": float(np.mean(hsv_px[:, 1])), "mean_V": float(np.mean(hsv_px[:, 2])),
        "mean_L": float(np.mean(L)), "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b)),
        "yellowness_index": yellowness_index,
        "astm_yellowness": astm_yellowness,
        "texture_std": texture_std,
        "roi_pixel_count": int(np.count_nonzero(roi_mask)),
    }
    return features


FEATURE_ORDER = [
    "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B",
    "mean_H", "mean_S", "mean_V", "mean_L", "mean_a", "mean_b", "std_b",
    "yellowness_index", "astm_yellowness", "texture_std",
]


def features_to_vector(features: dict) -> np.ndarray:
    """Deterministic ordering of the feature dict into a model-ready vector."""
    return np.array([features[k] for k in FEATURE_ORDER], dtype=np.float32)


if __name__ == "__main__":
    demo = np.full((100, 100, 3), (150, 220, 230), dtype=np.uint8)  # yellow-ish patch (BGR)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    feats = extract_features(demo, mask)
    for k, v in feats.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
