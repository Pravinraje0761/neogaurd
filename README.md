# NeoGuard: Newborn Jaundice Screening from Phone Photos (Research Prototype)

[![CI](https://github.com/Pravinraje0761/neogaurd/actions/workflows/ci.yml/badge.svg)](https://github.com/Pravinraje0761/neogaurd/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Not a diagnostic device.** This is a research prototype. Do not use it for clinical
> decisions; jaundice must be confirmed with a lab bilirubin test.

NeoGuard takes an ordinary phone photo of a newborn and returns a screening signal
(LOW / MONITOR / HIGH) for neonatal jaundice, with the reasons behind it and a recapture
prompt when the photo is unusable. A FastAPI backend serves both the model and a
clinician-facing web app.

## How it works

1. **Skin-region detection:** YCbCr skin-colour mask combined with a texture filter, with an
   optional face prior (`backend/src/real_data/skin_region_detection.py`).
2. **Quality gate:** rejects blurred, under- or over-exposed, or glare-affected captures and asks for a retake.
3. **Colour normalisation:** gray-world white balance, since phone photos have no colour
   reference card.
4. **Features:** colour and texture features from the skin region, including yellowness indices.
5. **Classifier:** a scikit-learn model (`backend/models/real_jaundice_classifier.joblib`)
   outputs a jaundice probability, mapped to LOW (< 0.33), MONITOR (< 0.66) or HIGH.

## Performance and limits

Trained on 760 real newborn photos. Single held-out split accuracy is about **75%**, and
sensitivity is about **50%** under 5-fold cross-validation, so it misses roughly half of
jaundiced cases. The API returns this caveat in a `disclaimer` field with every result.

- It outputs a jaundice probability, **not** a bilirubin level in mg/dL.
- The skin detector can lock onto incubator plastic when no face is in frame.
- `/api/analyze` has no authentication or rate limiting; add both before any public deployment.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --port 8000                   # open http://localhost:8000
```

Docker and hosting instructions (Render, Railway, Fly.io) are in [DEPLOY.md](DEPLOY.md).

## API

`POST /api/analyze` with a multipart `file` (image, up to 15 MB). The response `status` is
one of:

- `OK`: includes `risk_category`, `probability_jaundice`, `confidence`, `action`, `why`, `disclaimer`.
- `RECAPTURE_REQUIRED`: no confident skin region, or the photo failed the quality gate; `reasons` explains why.
- `ERROR`: unreadable or oversized file.

## Tests

```bash
pip install pytest httpx && pytest
```

Tests use synthetic images: an unreadable upload, a photo with no skin, and a full pass
through detection, quality checks, features and the bundled model. CI runs them on every push.

## License

Code: [MIT](LICENSE).
