"""
NeoGuard - Image Quality Control Module
-----------------------------------------
Implements the "Quality Check" stage of the NeoGuard AI pipeline (deck page 6, step 2):
blur detection, exposure detection, glare detection, and ROI validation.

Design principle (from pitch deck): if image quality is insufficient, NeoGuard must
refuse to predict and instead return "RECAPTURE REQUIRED" rather than force an
unreliable AI guess. This module is the gatekeeper that enforces that rule.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np


# ---- Tunable thresholds -----------------------------------------------------
# These are reasonable starting points for a prototype. In a real deployment they
# should be calibrated against a labeled dataset of "usable" vs "unusable" captures.
BLUR_VARIANCE_MIN = 60.0          # Variance of Laplacian; lower => blurrier
UNDEREXPOSED_MEAN_MAX = 40.0      # Mean pixel intensity (0-255) below this => too dark
# NOTE: the sclera is naturally near-white (high mean brightness) even in a good
# capture, so "overexposed" must be judged mainly by loss of DETAIL (a large
# fraction of truly clipped/saturated pixels), not by mean brightness alone.
OVEREXPOSED_MEAN_MIN = 252.0
OVEREXPOSED_CLIP_FRACTION = 0.35  # Fraction of pixels fully saturated (>=250) => overexposed
GLARE_HOT_PIXEL_FRACTION = 0.02   # Fraction of near-pure-white, near-zero-saturation pixels => specular glare
MIN_ROI_AREA_FRACTION = 0.01      # ROI (sclera/skin patch) must occupy at least this much of frame


@dataclass
class QualityReport:
    passed: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed:
            return "PASS - image quality sufficient for analysis"
        return "RECAPTURE REQUIRED - " + "; ".join(self.reasons)


def _blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian - a standard, fast focus/blur metric."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_stats(gray: np.ndarray, roi_mask: np.ndarray = None) -> dict:
    """Exposure is judged on the ROI when available (the region that actually
    drives the reading), falling back to the whole frame before ROI is known."""
    region = gray[roi_mask > 0] if (roi_mask is not None and np.count_nonzero(roi_mask) > 0) else gray
    mean_intensity = float(region.mean())
    clipped_high = float(np.mean(region >= 250))
    clipped_low = float(np.mean(region <= 5))
    return {
        "mean_intensity": mean_intensity,
        "clipped_high_fraction": clipped_high,
        "clipped_low_fraction": clipped_low,
    }


def _glare_score(bgr: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Detects specular highlights (glare): pixels that are very bright AND
    low-saturation (a wet/reflective highlight tends to wash out toward white,
    unlike genuinely yellow/bilirubin-tinted tissue which stays saturated).

    Restricted to the ROI when available -- glare on the surrounding face/skin
    doesn't affect the reading; glare *on the sclera itself* does.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    # Genuine specular highlights are near-PURE white (saturation collapses to
    # ~0), which is a materially stronger signal than merely "bright" -- a clean,
    # legitimately near-white sclera still carries some residual saturation.
    hot = (v > 250) & (s < 15)
    if roi_mask is not None and np.count_nonzero(roi_mask) > 0:
        region = roi_mask > 0
        return float(np.mean(hot[region]))
    return float(np.mean(hot))


def check_image_quality(bgr_image: np.ndarray, roi_mask: np.ndarray = None) -> QualityReport:
    """
    Run the full quality gate on a captured frame.

    Parameters
    ----------
    bgr_image : np.ndarray
        Captured frame, OpenCV BGR format, uint8.
    roi_mask : np.ndarray, optional
        Boolean/uint8 mask of the detected region of interest (sclera/skin patch).
        If provided, its area is checked against MIN_ROI_AREA_FRACTION.

    Returns
    -------
    QualityReport
    """
    if bgr_image is None or bgr_image.size == 0:
        return QualityReport(passed=False, reasons=["empty or unreadable image"])

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    reasons = []
    metrics = {}

    # 1) Blur / focus check
    blur = _blur_score(gray)
    metrics["blur_variance"] = blur
    if blur < BLUR_VARIANCE_MIN:
        reasons.append(f"image too blurry (focus score {blur:.1f} < {BLUR_VARIANCE_MIN})")

    # 2) Exposure check (ROI-restricted when a mask is available)
    exp = _exposure_stats(gray, roi_mask=roi_mask)
    metrics.update(exp)
    if exp["mean_intensity"] < UNDEREXPOSED_MEAN_MAX:
        reasons.append(f"image underexposed (mean intensity {exp['mean_intensity']:.1f})")
    elif exp["mean_intensity"] > OVEREXPOSED_MEAN_MIN or exp["clipped_high_fraction"] > OVEREXPOSED_CLIP_FRACTION:
        reasons.append("image overexposed / blown out highlights")

    # 3) Glare check (ROI-restricted when a mask is available)
    glare = _glare_score(bgr_image, roi_mask=roi_mask)
    metrics["glare_fraction"] = glare
    if glare > GLARE_HOT_PIXEL_FRACTION:
        reasons.append(f"specular glare detected ({glare*100:.1f}% of frame)")

    # 4) ROI validation (only if caller supplied a detected ROI mask)
    if roi_mask is not None:
        total_px = roi_mask.shape[0] * roi_mask.shape[1]
        roi_px = int(np.count_nonzero(roi_mask))
        roi_fraction = roi_px / total_px if total_px else 0.0
        metrics["roi_area_fraction"] = roi_fraction
        if roi_fraction < MIN_ROI_AREA_FRACTION:
            reasons.append("no valid sclera/skin region detected (ROI too small or missing)")

    return QualityReport(passed=(len(reasons) == 0), reasons=reasons, metrics=metrics)


if __name__ == "__main__":
    # Minimal self-test with a synthetic blank frame
    test_img = np.full((480, 640, 3), 128, dtype=np.uint8)
    report = check_image_quality(test_img)
    print(report.summary())
    print(report.metrics)
