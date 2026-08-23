# NeoGuard — wired app (mockup + real AI backend)

## What this is

- `static/index.html` — pitch-deck mockup, wired: "Capture" opens phone camera/file picker,
  uploads photo to `/api/analyze`, shows the real model result.
- `backend/app.py` — FastAPI server. Real-photo inference path (skin detection ->
  gray-world normalization -> color/texture features -> `real_jaundice_classifier.joblib`),
  NOT the synthetic calibrated-sclera pipeline from the original prototype. Outputs
  LOW/MONITOR/HIGH from a binary jaundice-probability classifier, not a mg/dL bilirubin
  estimate.
- `Dockerfile` — builds and serves both together on port 8000.

**Accuracy caveat (surfaced in the app's own `disclaimer` field):** trained on 760 real
newborn photos, single-split accuracy ~75%, sensitivity ~50% on 5-fold CV. Prototype /
research tool only, not a diagnostic device.

## Run locally

```
cd neoguard_app
python -m venv .venv
./.venv/Scripts/pip install -r backend/requirements.txt      # .venv/bin/pip on macOS/Linux
./.venv/Scripts/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — Guided capture tab, hit Capture, pick a photo.

## Deploy (any Docker host: Render / Railway / Fly.io / a VPS)

Repo root for the build is `neoguard_app/` (has the `Dockerfile`).

```
docker build -t neoguard-app .
docker run -p 8000:8000 neoguard-app
```

- **Render**: New → Web Service → connect repo → root directory `neoguard_app` →
  it auto-detects the Dockerfile → free tier works (sleeps when idle).
- **Railway**: New Project → Deploy from repo → root directory `neoguard_app` →
  Dockerfile auto-detected.
- **Fly.io**: `cd neoguard_app && fly launch` (it finds the Dockerfile), then `fly deploy`.

No environment variables or secrets needed — the model file ships in the image.

## Known gaps before this is more than a demo

- `real_jaundice_classifier.joblib` gives jaundice/normal probability, not calibrated
  bilirubin mg/dL — the hardware-calibrated path (`ai_prototype/src/pipeline.py` in the
  original zip) needs the physical color-reference attachment to produce that.
- Skin-region detector can mislock onto incubator plastic when no face is in frame
  (documented in the original `ai_prototype/README.md`).
- No auth/rate-limiting on `/api/analyze` — add before any public, non-demo deployment.
