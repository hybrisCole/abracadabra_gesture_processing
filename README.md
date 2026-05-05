# Gesture Recognition API

A FastAPI-based HTTP API for training and recognizing **atomic movements** (tap, wrist rotation, still) from IMU CSV data. It targets workflows aligned with a **Seeed XIAO nRF52840 Sense** companion pipeline; exported CSV **columns** match what this service expects (see below)—they are **not** identical to raw BLE-packed samples from the React Native app without conversion.

## Features

- Train a **Random Forest** classifier on hand-crafted **time + frequency** IMU features (`scikit-learn` + `scipy.signal`).
- Upload labeled CSV clips per movement type; **sliding-window** inference over longer recordings with smoothing and segment grouping.
- **Auto-learning**: high-/medium-confidence segments from `/api/predict` can be saved under `app/data/pending_training/` for confirm/reject via API or the atomic movement web UI.
- REST API with **Swagger** at `/docs` and **ReDoc** at `/redoc`.
- Browser forms: **Atomic Movement** (recommended), **Standard Upload** (legacy UI—see caveat below), **Model Details** (HTML dashboard).

## Important implementation notes

- **Sample rate assumption:** Feature extraction and sliding-window slicing assume **`fs = 250` Hz** (filters, Welch PSD, `/api/predict` window parameters). If your hardware exports **~200 Hz**, either resample on export or update `window_size_ms`, `overlap_ms`, and `fs` in `app/api/router.py` / `app/models/rf_gesture_model.py` so they stay consistent.
- **Active model:** **`RandomForestGestureModel`** in `app/models/rf_gesture_model.py`. **`app/models/gesture_model.py`** (DTW-based) is **legacy** and **not** wired into the API.
- **Deployment:** Training data and `rf_gesture_model.joblib` live under `app/data/` by default. On ephemeral hosts (Railway, Render, etc.), **redeploys can wipe uploads** unless you attach persistent storage—see `Dockerfile`, `render.yaml`, and `RAILWAY_DEPLOYMENT.md`.

## Project structure

```
.
├── app/
│   ├── main.py              # FastAPI app, CORS, HTML pages, /health, /api/transform-data
│   ├── api/
│   │   └── router.py        # /api/train, /api/predict, training CRUD, pending learning
│   ├── models/
│   │   ├── rf_gesture_model.py   # Random Forest (used)
│   │   └── gesture_model.py      # DTW (unused by router)
│   ├── utils/
│   │   └── data_handler.py   # CSV validation, training directory loader
│   └── data/
│       ├── training/         # Labeled CSV samples (e.g. tap_001.csv)
│       ├── pending_training/ # Auto-detected segments (runtime)
│       └── rf_gesture_model.joblib  # Saved model (after train)
├── pyproject.toml           # Python dependencies (source of truth for pip)
├── Dockerfile               # Production image (pinned deps, entrypoint for volume perms)
├── docker-entrypoint.sh     # chown /app/app/data then gosu → uvicorn
├── docker-compose.yml
├── render.yaml              # Render.com blueprint
├── Makefile
├── RAILWAY_DEPLOYMENT.md
├── GITHUB_SETUP.md
├── csv/                     # Optional reference CSVs
└── README.md
```

There is **no** `requirements.txt` at the repo root; use **`pyproject.toml`** (or install from the **Dockerfile** when containerizing).

## Installation

1. Clone the repository and enter the project directory (inner folder if your clone has a nested same-named directory):

   ```bash
   git clone <repository-url>
   cd abracadabra_gesture_processing
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies (editable install reads `pyproject.toml`; `scipy` is pulled in via `scikit-learn` for spectral features):

   ```bash
   pip install -e .
   ```

   For development tooling only:

   ```bash
   pip install -e ".[dev]"
   ```

## Usage

### Start the API server

```bash
uvicorn app.main:app --reload
```

- API root: http://localhost:8000  
- OpenAPI: http://localhost:8000/docs  
- Health (for load balancers): http://localhost:8000/health  

### Web interfaces

| URL | Purpose |
|-----|---------|
| `/` | JSON index with endpoint hints |
| `/atomic-movement-form` | Upload labeled atomic clips, test one window, analyze full recording, pending-review UI |
| `/upload-form` | **Legacy** upload/predict UI; `/api/predict` returns **multi-segment** sliding-window JSON (`significant_movements`, `detailed_segments`, …), **not** the older single-field `predicted_gesture` shape. Prefer the atomic form + `/api/predict-window` for single-window tests. |
| `/model-details` | HTML page; loads metrics from **`GET /api/model-details`** |

### API endpoints

All router endpoints are mounted under **`/api`** (see `app/main.py`).

**Training & model**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload-training-data` | Body: form `csv_data`, optional `movement_type` (`tap`, `wrist_rotation`, `still`). Saves CSV under `app/data/training/`. |
| POST | `/api/train` | Background train on all CSVs in `training/`; writes `app/data/rf_gesture_model.joblib`. |
| GET | `/api/model-status` | Trained or not; labels; optional CV summary. |
| GET | `/api/model-details` | JSON: feature count, movements, top feature importances, cross-validation. |
| GET | `/api/training-data` | List movements and sample counts. |
| DELETE | `/api/training-data/{movement_name}` | Remove CSVs for one movement prefix. |
| DELETE | `/api/delete-all-training-data` | Remove all training CSVs. |

**Prediction**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/predict-window` | Form `csv_data`: classify **one** short window → `predicted_movement`, `confidence`, `all_probabilities`. |
| POST | `/api/predict` | Form `csv_data`: **full recording** (typical use: multi-second clip, e.g. ~4 s device capture). Sliding windows (**350 ms**, **250 ms overlap**, **250 Hz** assumed), smoothing, segment merge, optional tap-split heuristics, `auto_learning` metadata. |

**Auto-learning (pending segments)**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pending-training-data` | List pending JSON metadata + linked CSV slices. |
| POST | `/api/confirm-detection/{detection_id}` | Form `correct_movement`: promote slice into `training/`. |
| DELETE | `/api/reject-detection/{detection_id}` | Mark rejected; does not add to training. |
| POST | `/api/auto-retrain` | Background retrain if enough confirmed pending items (see router logic). |

**Other**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/transform-data` | Form `csv_data`: map columns `rel_timestamp`/`acc_*`/`gyro_*` → `timeline`/`accX`/… CSV text in JSON (helper for alternate export formats). |

### CSV data format

Required columns for upload and prediction:

- `rel_timestamp` — relative time (ms) within the clip  
- `recording_id` — stable string id per recording (used when `movement_type` is omitted on upload)  
- `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z` — floats in **consistent physical or raw units** across train and inference  

Minimum row count and NaN rules are enforced in `app/utils/data_handler.py`.

## Workflow for atomic movement detection

1. **Collect training samples** (~350 ms tap, ~500 ms wrist rotation, ~300–500 ms still) as CSV with the columns above; use **`/atomic-movement-form`** to paste and label uploads.

2. **Train:** After enough examples per class (e.g. 20–30 each):

   ```bash
   curl -X POST http://localhost:8000/api/train
   ```

   Training runs in the **background**; refresh **`/model-details`** or **`GET /api/model-status`** after a short wait.

3. **Test:** Use **`POST /api/predict-window`** or the atomic form’s “Test Single Movement Window” section.

4. **Segment long recordings:** **`POST /api/predict`** with a full CSV stream returns counts, **`detailed_segments`** (start/end/duration), **`still_phases`**, raw/smoothed window predictions, and **`auto_learning`** suggestions. Review pending items via **`GET /api/pending-training-data`** and confirm/reject as needed.

## How it works

1. Short labeled clips are concatenated per movement type and turned into **feature vectors** (stats + spectral bands on acc/gyro axes and magnitudes), with **bandpass filtering** assuming **250 Hz**.

2. A **`RandomForestClassifier`** (`class_weight='balanced'`) is trained with **`StandardScaler`**; model + scaler are persisted with **`joblib`**.

3. Long-stream detection applies **overlapping windows** (defaults in `router.py`), **median-style smoothing** of isolated labels, **run-length grouping**, significance thresholds (e.g. ≥ 2 windows), and optional **multi-tap heuristics** on long tap runs.

## Docker / cloud

- **Docker:** `docker build` using the root `Dockerfile`. The container starts as root briefly: **`docker-entrypoint.sh`** creates/chowns **`/app/app/data`** for volume mounts, then runs **uvicorn** as **`appuser`** via **`gosu`**. Railway/Render pass **`$PORT`** as usual.  
- **Render:** `render.yaml` defines a web service and health check.  
- **Railway:** See `RAILWAY_DEPLOYMENT.md` and `railway.toml` / `nixpacks.toml`.

For production, restrict **`CORSMiddleware`** in `app/main.py` (currently `allow_origins=["*"]` for development).

## License

See `pyproject.toml` (MIT).
