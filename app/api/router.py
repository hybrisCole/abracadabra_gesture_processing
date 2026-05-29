import os
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.models.rf_gesture_model import RandomForestGestureModel
from app.schemas.recording import (
    AnalyzeRecordingIn,
    LabeledRecordingWindowIn,
    RecordingWindowIn,
    VerifyGesturePasswordIn,
)
from app.utils.rn_recording import (
    delete_training_samples,
    list_training_samples,
    normalize_movement_type,
    read_training_data,
    recording_to_frame,
    save_training_sample,
    validate_imu_frame,
)
from app.utils.segment_resolve import (
    resolve_segments_by_precedence,
    segments_to_password_sequence,
)

router = APIRouter()
model = RandomForestGestureModel()

MODEL_PATH = "app/data/rf_gesture_model.joblib"
TRAINING_DATA_DIR = "app/data/training"

# Load model if it exists
if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)


@router.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train the movement classifier from JSON samples under app/data/training."""
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    def _train_model():
        training_data = read_training_data(TRAINING_DATA_DIR)
        if not training_data:
            print("No training data found")
            return

        model.train(training_data)
        model.save(MODEL_PATH)
        print("Model training completed and saved")

    background_tasks.add_task(_train_model)
    return {"message": "Model training started", "status": "processing"}


def _require_trained() -> None:
    if not model.trained:
        raise HTTPException(status_code=400, detail="Model not trained yet")


def _frame_from_recording(recording: RecordingWindowIn):
    df, metadata = recording_to_frame(recording)
    is_valid, error_message = validate_imu_frame(df)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    return df, metadata


def _smooth_predictions(predictions: List[str]) -> List[str]:
    smoothed = predictions.copy()
    if len(predictions) > 2:
        for i in range(1, len(predictions) - 1):
            if (
                predictions[i] != predictions[i - 1]
                and predictions[i] != predictions[i + 1]
                and predictions[i - 1] == predictions[i + 1]
            ):
                smoothed[i] = predictions[i - 1]
    return smoothed


def _segments_from_windows(
    window_results: Dict[str, Any],
    *,
    min_confidence: float,
    min_segment_windows: int,
    include_still: bool,
) -> List[Dict[str, Any]]:
    predictions = window_results["window_predictions"]
    confidences = window_results["window_confidences"]
    starts = window_results["window_start_times"]
    ends = window_results["window_end_times"]
    centers = window_results["window_times"]

    if not predictions:
        return []

    smoothed = _smooth_predictions(predictions)
    segments: List[Dict[str, Any]] = []
    group_start = 0

    for i in range(1, len(smoothed) + 1):
        if i < len(smoothed) and smoothed[i] == smoothed[group_start]:
            continue

        label = smoothed[group_start]
        group_confidences = confidences[group_start:i]
        avg_confidence = float(sum(group_confidences) / len(group_confidences))
        window_count = i - group_start

        if (
            label != "error"
            and avg_confidence >= min_confidence
            and window_count >= min_segment_windows
            and (include_still or label != "still")
        ):
            start_ms = float(starts[group_start])
            end_ms = float(ends[i - 1])
            segments.append(
                {
                    "movement_type": label,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "center_ms": float(centers[group_start]),
                    "duration_ms": max(0.0, end_ms - start_ms),
                    "confidence": avg_confidence,
                    "window_count": window_count,
                }
            )
        group_start = i

    return segments


def _analyze_recording(payload: AnalyzeRecordingIn) -> Dict[str, Any]:
    _require_trained()
    df, metadata = _frame_from_recording(payload)
    window_results = model.predict_window_sequence(
        df,
        window_size_ms=payload.window_size_ms,
        overlap_ms=payload.overlap_ms,
        sample_rate_hz=metadata["sample_rate_hz"],
    )

    if not window_results["success"]:
        raise HTTPException(status_code=400, detail=window_results.get("error", "Window prediction failed"))

    segments = _segments_from_windows(
        window_results,
        min_confidence=payload.min_confidence,
        min_segment_windows=payload.min_segment_windows,
        include_still=payload.include_still,
    )
    resolved_segments = resolve_segments_by_precedence(segments)
    sequence = segments_to_password_sequence(resolved_segments)

    counts: Dict[str, int] = {}
    for segment in segments:
        movement = segment["movement_type"]
        counts[movement] = counts.get(movement, 0) + 1

    resolved_counts: Dict[str, int] = {}
    for segment in resolved_segments:
        movement = segment["movement_type"]
        resolved_counts[movement] = resolved_counts.get(movement, 0) + 1

    return {
        "recording_id": metadata["recording_id"],
        "window_id": metadata["window_id"],
        "sample_count": metadata["sample_count"],
        "sample_rate_hz": metadata["sample_rate_hz"],
        "duration_ms": metadata["duration_ms"],
        "counts": counts,
        "resolved_counts": resolved_counts,
        "segments": segments,
        "resolved_segments": resolved_segments,
        "sequence": sequence,
        "raw_window_predictions": {
            "predictions": window_results["window_predictions"],
            "smoothed_predictions": _smooth_predictions(window_results["window_predictions"]),
            "confidences": window_results["window_confidences"],
            "center_ms": window_results["window_times"],
            "start_ms": window_results["window_start_times"],
            "end_ms": window_results["window_end_times"],
        },
        "window_params": {
            "window_size_ms": payload.window_size_ms,
            "overlap_ms": payload.overlap_ms,
            "sample_rate_hz": metadata["sample_rate_hz"],
            "min_confidence": payload.min_confidence,
            "min_segment_windows": payload.min_segment_windows,
            "include_still": payload.include_still,
        },
    }


@router.post("/recordings/classify")
async def classify_recording_window(payload: RecordingWindowIn):
    """Classify one cropped gesture window from the RN app."""
    _require_trained()
    df, metadata = _frame_from_recording(payload)
    result = model.predict(df, sample_rate_hz=metadata["sample_rate_hz"])
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))

    return {
        "recording_id": metadata["recording_id"],
        "window_id": metadata["window_id"],
        "sample_count": metadata["sample_count"],
        "sample_rate_hz": metadata["sample_rate_hz"],
        "duration_ms": metadata["duration_ms"],
        "predicted_movement": result["predicted_movement"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
    }


@router.post("/recordings/analyze")
async def analyze_recording(payload: AnalyzeRecordingIn):
    """Analyze a full recording and return timed movement segments."""
    return _analyze_recording(payload)


@router.post("/gesture-passwords/verify")
async def verify_gesture_password(payload: VerifyGesturePasswordIn):
    """Compare precedence-resolved gesture sequence with an expected password."""
    analysis = _analyze_recording(payload)
    expected = [e.model_dump() for e in payload.expected_sequence]
    expected_movements = [normalize_movement_type(e["movement_type"]) for e in expected]
    detected_movements = analysis["sequence"]
    matched = detected_movements == expected_movements

    return {
        "matched": matched,
        "expected_sequence": expected_movements,
        "detected_sequence": detected_movements,
        "analysis": analysis,
    }


@router.post("/training-samples")
async def upload_training_sample(payload: LabeledRecordingWindowIn):
    """Persist a labeled RN crop as JSON training data."""
    try:
        saved = save_training_sample(TRAINING_DATA_DIR, payload)
        return {"message": "Training sample saved", **saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.get("/model-status")
async def get_model_status():
    """
    Get the current status of the atomic movement recognition model.
    """
    if not model.trained:
        return {"status": "not_trained", "message": "Model not trained yet"}
    
    # For Random Forest model we have more detailed information
    response = {
        "status": "trained",
        "model_type": "RandomForest",
        "movements": list(model.gesture_labels),
        "num_movements": len(model.gesture_labels)
    }
    
    # Add cross-validation information if available
    if model.cross_val_scores is not None:
        response["cross_validation"] = {
            "scores": model.cross_val_scores.tolist(),
            "mean_accuracy": float(model.cross_val_scores.mean()),
            "std_accuracy": float(model.cross_val_scores.std())
        }
    
    return response

@router.get("/model-details")
async def get_model_details():
    """
    Get detailed information about the trained model, including feature importances.
    """
    if not model.trained:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Get top features by importance
    top_features = dict(sorted(model.feature_importances.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:30])  # Top 30 features
    
    return {
        "model_type": "RandomForest",
        "num_features": len(model.feature_names),
        "num_movements": len(model.gesture_labels),
        "movements": list(model.gesture_labels),
        "top_features": top_features,
        "cross_validation": {
            "scores": model.cross_val_scores.tolist() if model.cross_val_scores is not None else None,
            "mean_accuracy": float(model.cross_val_scores.mean()) if model.cross_val_scores is not None else None,
            "std_accuracy": float(model.cross_val_scores.std()) if model.cross_val_scores is not None else None
        }
    }

@router.get("/training-samples")
async def get_training_samples():
    """List JSON training samples currently stored on the volume."""
    return list_training_samples(TRAINING_DATA_DIR)


@router.delete("/training-samples/{movement_type}")
async def delete_training_samples_for_movement(movement_type: str):
    """Delete all JSON samples for one movement label."""
    try:
        deleted_count = delete_training_samples(TRAINING_DATA_DIR, movement_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No samples found for '{movement_type}'")
    return {"message": f"Deleted {deleted_count} samples", "deleted_count": deleted_count}


@router.delete("/training-samples")
async def delete_all_training_samples():
    """Delete all JSON training samples."""
    deleted_count = delete_training_samples(TRAINING_DATA_DIR)
    return {"message": f"Deleted {deleted_count} samples", "deleted_count": deleted_count}