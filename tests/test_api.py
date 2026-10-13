"""API tests for the NeoGuard screening prototype (no network, synthetic images)."""
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import DISCLAIMER, app, get_model  # noqa: E402

client = TestClient(app)


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return {"file": ("photo.png", buf.tobytes(), "image/png")}


def _skin_photo(seed=0):
    """A textured skin-tone patch (~40% of the frame) on a dark cloth background (BGR).

    The detector rejects frames that are almost entirely "skin", so the patch must not
    fill the image.
    """
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(45, 8, (480, 640, 3)), 0, 255).astype(np.float32)
    patch = np.full((300, 400, 3), (120, 160, 210), dtype=np.float32)
    patch += rng.normal(0, 7, patch.shape)
    img[90:390, 120:520] = patch
    return np.clip(img, 0, 255).astype(np.uint8)


def test_homepage_is_served():
    assert client.get("/").status_code == 200


def test_unreadable_upload_is_rejected():
    r = client.post("/api/analyze", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.json()["status"] == "ERROR"


def test_photo_without_skin_asks_for_a_recapture():
    blue = np.zeros((480, 640, 3), dtype=np.uint8)
    blue[:] = (200, 60, 20)
    r = client.post("/api/analyze", files=_png(blue))
    assert r.json()["status"] == "RECAPTURE_REQUIRED"


def test_bundled_model_loads_and_scores():
    clf = get_model()
    assert hasattr(clf, "predict_proba")


def test_full_pipeline_returns_a_bounded_risk_with_the_disclaimer():
    body = client.post("/api/analyze", files=_png(_skin_photo())).json()
    assert body["status"] == "OK", body.get("reasons")
    assert body["risk_category"] in {"LOW", "MONITOR", "HIGH"}
    assert 0.0 <= body["probability_jaundice"] <= 1.0
    assert body["disclaimer"] == DISCLAIMER
    assert "NOT a diagnostic device" in body["disclaimer"]
