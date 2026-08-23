"""
NeoGuard - API server
----------------------
Wires the real-photo inference path (src/real_data/*, models/real_jaundice_classifier.joblib)
to an HTTP endpoint the app mockup's "Capture" screen calls, and serves the mockup itself
as the app's static frontend.

This is the REAL-PHOTO path (uncontrolled phone camera photo, no calibration target),
not the synthetic calibrated-sclera pipeline in ai_prototype/src/pipeline.py. It outputs
a binary jaundiced/normal probability from src/real_data, not a calibrated bilirubin
mg/dL estimate -- see backend/README_DEPLOY.md and the model's own training report
(reports/real_data_metrics_report.json in the original prototype) for accuracy caveats
this API's disclaimer field is built from.
"""

import os
import sys

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "real_data"))

from feature_extraction import extract_features, features_to_vector   # noqa: E402
from image_quality import check_image_quality                         # noqa: E402
from skin_region_detection import detect_skin_region                  # noqa: E402
from uncalibrated_normalization import gray_world_normalize            # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "real_jaundice_classifier.joblib")
STATIC_DIR = os.path.join(os.path.dirname(ROOT), "static")
MAX_UPLOAD_MB = 15
MIN_SKIN_CONFIDENCE = 0.25

DISCLAIMER = (
    "Prototype research tool, NOT a diagnostic device. Trained/validated on a "
    "small real-photo dataset (760 images, single held-out split accuracy ~75%, "
    "sensitivity ~50% by 5-fold cross-validation -- see project README). Do not "
    "use for clinical decisions. Always confirm with a lab bilirubin test."
)

RISK_ACTION = {
    "LOW": "No jaundice signal detected in this photo. Continue routine newborn monitoring.",
    "MONITOR": "Ambiguous signal. Recommend a follow-up visual check or repeat capture in better light.",
    "HIGH": "Jaundice-consistent signal detected. Recommend prompt clinical bilirubin (lab TSB/TcB) follow-up.",
}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


app = FastAPI(title="NeoGuard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_model():
    get_model()


def _risk_category(prob_jaundice: float) -> str:
    if prob_jaundice < 0.33:
        return "LOW"
    if prob_jaundice < 0.66:
        return "MONITOR"
    return "HIGH"


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return {"status": "ERROR", "reasons": [f"file exceeds {MAX_UPLOAD_MB}MB limit"]}

    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "ERROR", "reasons": ["unreadable image file"]}

    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    mask, roi_confidence = detect_skin_region(img)
    if roi_confidence < MIN_SKIN_CONFIDENCE:
        return {
            "status": "RECAPTURE_REQUIRED",
            "reasons": ["no confident newborn skin region detected in this photo"],
            "roi_confidence": round(float(roi_confidence), 3),
        }

    quality_report = check_image_quality(img, roi_mask=mask)
    if not quality_report.passed:
        return {
            "status": "RECAPTURE_REQUIRED",
            "reasons": quality_report.reasons,
            "quality_metrics": {k: round(float(v), 3) for k, v in quality_report.metrics.items()},
            "roi_confidence": round(float(roi_confidence), 3),
        }

    normalized = gray_world_normalize(img)
    try:
        features = extract_features(normalized, mask)
    except ValueError as e:
        return {"status": "RECAPTURE_REQUIRED", "reasons": [f"feature extraction failed: {e}"]}

    vec = features_to_vector(features).reshape(1, -1)
    clf = get_model()
    prob_jaundice = float(clf.predict_proba(vec)[0, 1])
    risk = _risk_category(prob_jaundice)
    confidence = float(max(prob_jaundice, 1 - prob_jaundice))

    why = []
    if risk in ("MONITOR", "HIGH"):
        if features["yellowness_index"] > 5:
            why.append("Elevated yellowness index")
        if features["astm_yellowness"] > 5:
            why.append("Elevated ASTM yellowness")
        if not why:
            why.append("Model signal above baseline (color features near threshold)")
    else:
        why.append("Yellowness index near baseline for detected skin region")
    if roi_confidence > 0.6:
        why.append("High-confidence skin region")

    return {
        "status": "OK",
        "risk_category": risk,
        "probability_jaundice": round(prob_jaundice, 3),
        "confidence": round(confidence, 3),
        "action": RISK_ACTION[risk],
        "roi_confidence": round(float(roi_confidence), 3),
        "why": why,
        "disclaimer": DISCLAIMER,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
