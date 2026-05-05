# Abracadabra Gesture Processing

JSON-only FastAPI service for training and detecting timed gesture sequences from **abracadabra-rnapp** IMU recordings. This service is the ML/backend piece of the Abracadabra gesture-password flow:

```text
abracadabra-platformio  ->  abracadabra-rnapp  ->  abracadabra_gesture_processing
records raw IMU             receives/crops          trains + detects timed gestures
```

The previous prototype has been replaced. The public API accepts the same shape the React Native app already has after BLE decode: `window_id` plus `samples[]` with `t_ms`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`.

## What It Does

- Stores labeled gesture crops as JSON under `app/data/training/<movement_type>/`.
- Trains a Random Forest classifier on hand-crafted IMU features from raw axes and derived magnitudes.
- Classifies one cropped gesture window.
- Analyzes a full 3-4 second recording by sliding windows across the timeline and returning timed segments.
- Optionally compares detected non-still segments to an expected gesture-password sequence.

Supported movement labels:

- `tap`
- `double_tap`
- `still` (alias: `silence`)
- `wrist_rotation`

## JSON Contract

Training/classification payloads use this shape:

```json
{
  "window_id": 17,
  "recording_id": "optional-client-id",
  "sample_rate_hz": 200,
  "samples": [
    {"t_ms": 0, "ax": 120, "ay": -40, "az": 1024, "gx": 8, "gy": -3, "gz": 1},
    {"t_ms": 5, "ax": 122, "ay": -39, "az": 1021, "gx": 9, "gy": -4, "gz": 2}
  ]
}
```

For labeled training crops, add `movement_type`:

```json
{
  "movement_type": "tap",
  "window_id": 17,
  "samples": [
    {"t_ms": 1250, "ax": 120, "ay": -40, "az": 1024, "gx": 8, "gy": -3, "gz": 1}
  ]
}
```

`t_ms`, `window_id`, and `recording_id` are metadata/timing only; they are not ML features. The model learns from `ax`...`gz` and derived features. Cropped windows are normalized to crop-relative time internally so a tap at 500 ms and a tap at 3200 ms can train the same class.

## API Endpoints

All API routes are mounted under `/api`.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/training-samples` | Save one labeled RN crop as JSON training data. |
| `GET` | `/api/training-samples` | List stored labels, sample counts, and metadata. |
| `DELETE` | `/api/training-samples/{movement_type}` | Delete all samples for one movement. |
| `DELETE` | `/api/training-samples` | Delete all training samples. |
| `POST` | `/api/train` | Train the Random Forest from stored JSON samples and save `rf_gesture_model.joblib`. |
| `GET` | `/api/model-status` | Report trained/not trained, labels, and cross-validation summary. |
| `GET` | `/api/model-details` | Return feature count, labels, top feature importances, and CV stats. |
| `POST` | `/api/recordings/classify` | Classify one cropped gesture window. |
| `POST` | `/api/recordings/analyze` | Analyze a full recording and return timed gesture segments. |
| `POST` | `/api/gesture-passwords/verify` | Analyze a recording and compare non-still segment labels with an expected sequence. |

Health/documentation:

- `GET /`
- `GET /health`
- `GET /docs`
- `GET /redoc`

## Full Recording Analysis

`POST /api/recordings/analyze` is for 3-4 second recordings that may contain multiple events:

```text
still -> tap -> still -> double_tap -> still -> wrist_rotation
```

The service sorts samples by `t_ms`, infers sample rate from median `t_ms` deltas unless provided, slides an overlapping window across the recording, classifies each window, smooths isolated one-window outliers, and merges adjacent windows into timed segments.

Example response shape:

```json
{
  "segments": [
    {"movement_type": "tap", "start_ms": 420, "end_ms": 760, "confidence": 0.91},
    {"movement_type": "double_tap", "start_ms": 1180, "end_ms": 1760, "confidence": 0.88},
    {"movement_type": "wrist_rotation", "start_ms": 2460, "end_ms": 3120, "confidence": 0.93}
  ],
  "counts": {"tap": 1, "double_tap": 1, "wrist_rotation": 1}
}
```

## Training Workflow

1. Use `abracadabra-rnapp` to receive a recording from the wearable.
2. Crop a gesture window in the RN timeline.
3. Upload the crop to `POST /api/training-samples` with `movement_type`.
4. Repeat for enough examples of `tap`, `double_tap`, `still`, and `wrist_rotation`.
5. Call `POST /api/train`.
6. Send full recordings to `POST /api/recordings/analyze`.
7. Add more labeled crops and retrain when detection misses or confuses gestures.

The volume is empty on first Railway mount; initial training data must be uploaded as JSON from the app or seeded manually as JSON files.

## Project Structure

```text
.
├── app/
│   ├── main.py                    # FastAPI app, CORS, root, /health
│   ├── api/router.py              # JSON-only API routes
│   ├── schemas/recording.py       # RN-shaped Pydantic request models
│   ├── utils/rn_recording.py      # JSON persistence + DataFrame conversion
│   ├── models/rf_gesture_model.py # Random Forest feature pipeline
│   └── data/
│       ├── training/              # JSON labeled crops on Railway volume
│       ├── pending_training/      # Reserved for future review workflows
│       └── rf_gesture_model.joblib
├── Dockerfile
├── docker-entrypoint.sh
├── pyproject.toml
├── railway.toml
└── RAILWAY_DEPLOYMENT.md
```

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

Open:

- http://localhost:8000/docs
- http://localhost:8000/health

## Railway

Mount the Railway volume at:

```text
/app/app/data
```

`docker-entrypoint.sh` starts as root briefly, creates/chowns the mounted data folders for `appuser`, then starts uvicorn as `appuser`.

Railway sets `PORT`; optional vars:

- `API_TITLE`
- `API_VERSION`
