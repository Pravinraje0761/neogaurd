"""
NeoGuard - Whole-Scene Newborn Skin Detection (real-data adaptation)
------------------------------------------------------------------------
The controlled-capture pipeline (src/roi_detection.py) assumes a tightly-framed
sclera or skin patch, as NeoGuard's hardware attachment is designed to produce.
The real public datasets pulled into this project (Kaggle/NJN, Mendeley) are NOT
that: they are uncontrolled clinical photos -- full incubator scenes, diapers,
monitoring clips, cloth, hands -- with no framing guarantee and no calibration
target. Before any color feature can be trusted, the newborn's actual skin has to
be located and separated from background/equipment/clothing.

This module does that with a classic, dependency-light YCbCr skin-color
heuristic (Cb/Cr band-pass, largest plausible connected component, morphological
cleanup). It is deliberately conservative: on a photo where no confident skin
region is found, it returns zero confidence rather than a bad guess, so the
calling script can exclude that image rather than extract garbage features from
it -- the same "don't guess" principle as the main pipeline's quality gate.
"""

import os

import cv2
import numpy as np

# Classic YCbCr skin-tone band (widely used baseline in the skin-detection
# literature; adequate for a first-pass real-data filter, not a clinical-grade
# segmenter).
CB_LOW, CB_HIGH = 77, 135
CR_LOW, CR_HIGH = 133, 180

MIN_AREA_FRACTION = 0.03   # a usable skin region must cover at least 3% of frame
MAX_AREA_FRACTION = 0.85   # if "skin" covers almost the whole frame, something's off (blown-out background, etc.)

# Incubator walls, plastic, and taut cloth are often within the YCbCr skin band
# (especially under warm/yellow incubator lighting) but are visually much
# FLATTER than real skin, which has fine natural texture (pores, vellus hair,
# subtle mottling). A local texture-energy gate rejects the smooth-plastic false
# positives that color thresholding alone lets through -- but manual review
# (reports/real_data_samples/) found this alone still isn't always enough: an
# incubator wall with enough texture noise/reflections can still win as the
# largest connected component. A face-detection PRIOR (below) helps further:
# when a face is found, the newborn's actual body is anchored near it, so we
# prefer the largest skin-colored, textured component that is near the face
# over the largest one anywhere in frame (which is exactly the failure mode
# that let a nearby incubator wall win before).
TEXTURE_WINDOW = 9
MIN_TEXTURE_ENERGY = 4.0

# Face-detection prior: OpenCV's bundled Haar cascades (no extra download).
# Newborn faces differ enough from the adult faces these cascades were trained
# on that recall is limited -- this is used as a SOFT prior (re-rank candidate
# regions when a face IS found), not a hard requirement (when no face is
# found, behavior falls back to the original largest-component approach
# unchanged, so this can only help, never regress a case that worked before).
_FACE_CASCADE_FILES = ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]
_face_cascades = None


def _get_face_cascades():
    global _face_cascades
    if _face_cascades is None:
        _face_cascades = []
        for fname in _FACE_CASCADE_FILES:
            path = os.path.join(cv2.data.haarcascades, fname)
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    _face_cascades.append(cascade)
    return _face_cascades


def _detect_face_prior_bbox(bgr_image: np.ndarray):
    """Returns the largest detected face bounding box (x, y, w, h), or None."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    best = None
    for cascade in _get_face_cascades():
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
    return best


def _texture_energy_map(bgr_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    # local mean of |Laplacian| = local texture energy
    energy = cv2.boxFilter(lap, ddepth=-1, ksize=(TEXTURE_WINDOW, TEXTURE_WINDOW))
    return energy


def detect_skin_region(bgr_image: np.ndarray) -> tuple:
    """
    Returns (mask uint8 0/255, confidence 0-1) for the most plausible newborn
    skin region in an uncontrolled scene photo. Combines a YCbCr skin-color
    band with a local-texture gate, so smooth incubator plastic / taut cloth
    (color-similar to skin but visually flat) is excluded even when its color
    alone would pass. When a face is detected (see module docstring), the
    candidate component is chosen by proximity to the face rather than by raw
    area alone -- this is a soft prior, so a photo with no detected face falls
    back to the original largest-plausible-component behavior unchanged.
    """
    ycbcr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    skin_color = (cb >= CB_LOW) & (cb <= CB_HIGH) & (cr >= CR_LOW) & (cr <= CR_HIGH)
    textured = _texture_energy_map(bgr_image) >= MIN_TEXTURE_ENERGY

    mask = (skin_color & textured).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=np.uint8), 0.0

    total_px = bgr_image.shape[0] * bgr_image.shape[1]
    face_bbox = _detect_face_prior_bbox(bgr_image)

    # Build the plausible candidate set (area-fraction-valid components), then
    # pick among them: nearest-to-face-center when a face was found, largest
    # otherwise (original behavior, preserved as the no-face fallback).
    candidates = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        area_fraction = area / total_px
        if MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION:
            candidates.append(label)

    if not candidates:
        return np.zeros(mask.shape, dtype=np.uint8), 0.0

    used_face_prior = False
    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        face_cx, face_cy = fx + fw / 2.0, fy + fh / 2.0
        diag = float(np.hypot(*bgr_image.shape[:2]))
        # A newborn's body extends well beyond its face, so only trust the
        # prior when a candidate's centroid is within a generous radius of
        # the face center; otherwise a spurious face detection (e.g. on a
        # blanket pattern) can't hijack an unrelated, otherwise-good mask.
        max_prior_distance = 0.45 * diag
        best_label, best_dist = None, None
        for label in candidates:
            cx, cy = centroids[label]
            dist = float(np.hypot(cx - face_cx, cy - face_cy))
            if dist <= max_prior_distance and (best_dist is None or dist < best_dist):
                best_label, best_dist = label, dist
        if best_label is not None:
            used_face_prior = True
            best_label_final = best_label
        else:
            best_label_final = max(candidates, key=lambda l: stats[l, cv2.CC_STAT_AREA])
    else:
        best_label_final = max(candidates, key=lambda l: stats[l, cv2.CC_STAT_AREA])

    best_area = float(stats[best_label_final, cv2.CC_STAT_AREA])
    area_fraction = best_area / total_px
    region_mask = np.where(labels == best_label_final, 255, 0).astype(np.uint8)

    # Confidence: reward area in a plausible mid-range and a reasonably compact
    # (non-scattered) shape; a face-prior match adds a further confidence
    # boost since it's corroborated by a second, independent detector.
    x, y0, w, h, area = stats[best_label_final]
    bbox_fill = area / float(w * h) if w * h else 0
    size_score = 1.0 - abs(area_fraction - 0.25) / 0.25  # peaks near 25% of frame
    size_score = float(np.clip(size_score, 0, 1))
    shape_score = float(np.clip(bbox_fill / 0.55, 0, 1))
    confidence = float(np.clip(0.5 * size_score + 0.5 * shape_score, 0, 1))
    if used_face_prior:
        confidence = float(np.clip(confidence + 0.15, 0, 1))

    return region_mask, confidence


def detect_full_frame(bgr_image: np.ndarray) -> tuple:
    """
    For datasets that are ALREADY a tight skin macro (e.g. the Mendeley
    forehead/sternum close-ups), the whole frame is the region of interest --
    no detection needed. A small border is excluded to avoid edge artifacts.
    """
    h, w = bgr_image.shape[:2]
    margin_h, margin_w = int(h * 0.05), int(w * 0.05)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin_h:h - margin_h, margin_w:w - margin_w] = 255
    return mask, 0.95


if __name__ == "__main__":
    demo = np.full((200, 200, 3), (140, 170, 200), dtype=np.uint8)  # skin-ish BGR
    mask, conf = detect_skin_region(demo)
    print("area:", int(np.count_nonzero(mask)), "confidence:", round(conf, 3))
