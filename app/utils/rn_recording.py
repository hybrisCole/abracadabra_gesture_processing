import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from app.schemas.recording import ImuSampleIn, LabeledRecordingWindowIn, RecordingWindowIn


VALID_MOVEMENTS = {"tap", "double_tap", "still", "wrist_rotation"}
REQUIRED_FRAME_COLUMNS = [
    "rel_timestamp",
    "recording_id",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def normalize_movement_type(value: str) -> str:
    movement = value.strip().lower()
    movement = "still" if movement == "silence" else movement
    if movement not in VALID_MOVEMENTS:
        raise ValueError(f"Unsupported movement_type '{value}'")
    return movement


def _sample_dict(sample: ImuSampleIn) -> Dict[str, Any]:
    return sample.model_dump()


def infer_sample_rate_hz(samples: Iterable[ImuSampleIn]) -> float:
    ordered = sorted(samples, key=lambda s: s.t_ms)
    if len(ordered) < 2:
        return 200.0

    deltas = np.diff([s.t_ms for s in ordered])
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 200.0

    median_delta_ms = float(np.median(deltas))
    if median_delta_ms <= 0:
        return 200.0

    return float(np.clip(1000.0 / median_delta_ms, 50.0, 500.0))


def recording_to_frame(recording: RecordingWindowIn) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ordered = sorted(recording.samples, key=lambda s: s.t_ms)
    first_t_ms = ordered[0].t_ms
    last_t_ms = ordered[-1].t_ms
    recording_id = recording.recording_id or (
        f"window_{recording.window_id}" if recording.window_id is not None else f"recording_{uuid.uuid4().hex[:8]}"
    )
    sample_rate_hz = float(np.clip(recording.sample_rate_hz or infer_sample_rate_hz(ordered), 50.0, 500.0))

    rows = [
        {
            "rel_timestamp": float(sample.t_ms - first_t_ms),
            "recording_id": recording_id,
            "acc_x": float(sample.ax),
            "acc_y": float(sample.ay),
            "acc_z": float(sample.az),
            "gyro_x": float(sample.gx),
            "gyro_y": float(sample.gy),
            "gyro_z": float(sample.gz),
        }
        for sample in ordered
    ]

    metadata = {
        "recording_id": recording_id,
        "window_id": recording.window_id,
        "sample_count": len(ordered),
        "sample_rate_hz": sample_rate_hz,
        "original_start_ms": first_t_ms,
        "original_end_ms": last_t_ms,
        "duration_ms": last_t_ms - first_t_ms,
    }
    return pd.DataFrame(rows), metadata


def validate_imu_frame(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "Data is empty"

    missing = [col for col in REQUIRED_FRAME_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    if len(df) < 10:
        return False, f"Not enough data points: {len(df)} (minimum 10 required)"

    numeric_cols = [c for c in REQUIRED_FRAME_COLUMNS if c != "recording_id"]
    if df[numeric_cols].isna().any().any():
        return False, "Data contains NaN values"

    if not df["rel_timestamp"].is_monotonic_increasing:
        return False, "rel_timestamp must be monotonic increasing"

    return True, ""


def save_training_sample(data_dir: str, payload: LabeledRecordingWindowIn) -> Dict[str, Any]:
    movement_type = normalize_movement_type(payload.movement_type)
    df, metadata = recording_to_frame(payload)
    is_valid, error = validate_imu_frame(df)
    if not is_valid:
        raise ValueError(error)

    sample_id = uuid.uuid4().hex[:12]
    movement_dir = Path(data_dir) / movement_type
    movement_dir.mkdir(parents=True, exist_ok=True)
    file_path = movement_dir / f"{sample_id}.json"

    saved = {
        "sample_id": sample_id,
        "movement_type": movement_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "samples": [_sample_dict(sample) for sample in sorted(payload.samples, key=lambda s: s.t_ms)],
    }
    file_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    return {
        "sample_id": sample_id,
        "movement_type": movement_type,
        "path": str(file_path),
        "sample_count": metadata["sample_count"],
        "sample_rate_hz": metadata["sample_rate_hz"],
        "duration_ms": metadata["duration_ms"],
    }


def read_training_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Read JSON training samples. Each saved crop becomes one RF training example."""
    root = Path(data_dir)
    if not root.exists():
        return {}

    training: Dict[str, pd.DataFrame] = {}
    for movement_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            movement_type = normalize_movement_type(movement_dir.name)
        except ValueError:
            continue

        for file_path in sorted(movement_dir.glob("*.json")):
            try:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
                samples = [ImuSampleIn(**sample) for sample in raw.get("samples", [])]
                recording = RecordingWindowIn(
                    window_id=raw.get("window_id"),
                    recording_id=raw.get("recording_id") or raw.get("sample_id"),
                    sample_rate_hz=raw.get("sample_rate_hz"),
                    samples=samples,
                )
                df, _ = recording_to_frame(recording)
                is_valid, error = validate_imu_frame(df)
                if not is_valid:
                    print(f"Skipping {file_path}: {error}")
                    continue
                training[f"{movement_type}#{file_path.stem}"] = df
            except Exception as exc:
                print(f"Skipping {file_path}: {exc}")

    return training


def list_training_samples(data_dir: str) -> Dict[str, Any]:
    root = Path(data_dir)
    sample_counts: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []

    if not root.exists():
        return {"movements": [], "sample_counts": {}, "total_samples": 0, "samples": []}

    for movement_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            movement_type = normalize_movement_type(movement_dir.name)
        except ValueError:
            continue

        files = sorted(movement_dir.glob("*.json"))
        sample_counts[movement_type] = len(files)
        for file_path in files:
            try:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
                samples.append(
                    {
                        "sample_id": raw.get("sample_id") or file_path.stem,
                        "movement_type": movement_type,
                        "sample_count": raw.get("sample_count", len(raw.get("samples", []))),
                        "sample_rate_hz": raw.get("sample_rate_hz"),
                        "duration_ms": raw.get("duration_ms"),
                        "created_at": raw.get("created_at"),
                    }
                )
            except Exception:
                samples.append({"sample_id": file_path.stem, "movement_type": movement_type})

    return {
        "movements": sorted(sample_counts.keys()),
        "sample_counts": sample_counts,
        "total_samples": sum(sample_counts.values()),
        "samples": samples,
    }


def delete_training_samples(data_dir: str, movement_type: str | None = None) -> int:
    root = Path(data_dir)
    if not root.exists():
        return 0

    if movement_type is None:
        count = sum(1 for _ in root.rglob("*.json"))
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file() and child.suffix == ".json":
                child.unlink()
        return count

    movement = normalize_movement_type(movement_type)
    movement_dir = root / movement
    if not movement_dir.exists():
        return 0

    count = sum(1 for _ in movement_dir.glob("*.json"))
    shutil.rmtree(movement_dir)
    return count
