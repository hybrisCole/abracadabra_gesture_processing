from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import os
import pandas as pd
from pathlib import Path
import io

from app.models.rf_gesture_model import RandomForestGestureModel
from app.utils.data_handler import read_csv_from_bytes, read_training_data, validate_imu_data

router = APIRouter()
model = RandomForestGestureModel()

MODEL_PATH = "app/data/rf_gesture_model.joblib"
TRAINING_DATA_DIR = "app/data/training"

# Load model if it exists
if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH)

@router.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """
    Train the atomic movement recognition model using the data in the training directory.
    The training happens in the background.
    
    The model will be trained to recognize short, atomic movements like 'tap', 'wrist_rotation',
    or 'still' from IMU data. These movements can then be detected within longer streams of data.
    """
    # Ensure training data directory exists
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    def _train_model():
        # Read training data
        training_data = read_training_data(TRAINING_DATA_DIR)
        
        if not training_data:
            print("No training data found")
            return
        
        # Train the model
        model.train(training_data)
        
        # Save the model
        model.save(MODEL_PATH)
        print("Model training completed and saved")
    
    # Start training in the background
    background_tasks.add_task(_train_model)
    
    return {"message": "Model training started", "status": "processing"}

@router.post("/predict")
async def predict_gesture(csv_data: str = Form(...)):
    """
    Detect and count atomic movements (like taps and wrist rotations) in a longer data stream.
    
    Uses a sliding window approach to identify each type of movement and counts occurrences.
    Also identifies periods of non-movement ("still" phases).
    
    Args:
        csv_data: CSV text data containing IMU readings (typically 5 seconds of data)
        
    Returns:
        JSON with counts of each detected movement type, confidence scores, and timing information
    """
    if not model.trained:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        # Convert string data to bytes and read it
        content = csv_data.encode('utf-8')
        df = read_csv_from_bytes(content)
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Could not parse CSV data. Make sure it's valid CSV format.")
        
        is_valid, error_message = validate_imu_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Use sliding window to analyze the entire data stream
        # Default to 350ms windows with 100ms step (250ms overlap)
        window_size_ms = 350  
        overlap_ms = 250
        sample_rate_hz = 250  # Assumed from original code
        
        # Get window predictions using the sliding window approach
        window_results = model.predict_window_sequence(
            df, 
            window_size_ms=window_size_ms, 
            overlap_ms=overlap_ms, 
            sample_rate_hz=sample_rate_hz
        )
        
        if not window_results["success"]:
            raise HTTPException(status_code=500, detail=window_results.get("error", "Window prediction failed"))
        
        # Post-process window predictions to count movements and still phases
        window_preds = window_results["window_predictions"]
        window_confs = window_results["window_confidences"]
        window_times = window_results["window_times"]
        
        # Simple smoothing: Replace isolated predictions with their neighbors
        # if prediction is different from both previous and next, and both prev and next are the same
        smoothed_preds = window_preds.copy()
        if len(window_preds) > 2:
            for i in range(1, len(window_preds) - 1):
                if (window_preds[i] != window_preds[i-1] and 
                    window_preds[i] != window_preds[i+1] and 
                    window_preds[i-1] == window_preds[i+1]):
                    smoothed_preds[i] = window_preds[i-1]
        
        # Group consecutive identical predictions
        groups = []
        current_group = {"movement": smoothed_preds[0], "count": 1, "start_time": window_times[0]}
        
        for i in range(1, len(smoothed_preds)):
            if smoothed_preds[i] == current_group["movement"]:
                # Continue current group
                current_group["count"] += 1
            else:
                # End current group
                current_group["end_time"] = window_times[i-1]
                current_group["duration"] = current_group["end_time"] - current_group["start_time"]
                groups.append(current_group)
                
                # Start new group
                current_group = {"movement": smoothed_preds[i], "count": 1, "start_time": window_times[i]}
        
        # Add the last group
        if current_group:
            current_group["end_time"] = window_times[-1]
            current_group["duration"] = current_group["end_time"] - current_group["start_time"]
            groups.append(current_group)
        
        # Count each type of movement
        movement_counts = {}
        for group in groups:
            movement = group["movement"]
            movement_counts[movement] = movement_counts.get(movement, 0) + 1
        
        # Count significant movements (using minimum window threshold)
        # Only count a movement if it spans at least 2 windows (to avoid noise)
        significant_groups = [g for g in groups if g["count"] >= 2]
        significant_movements = {}
        for group in significant_groups:
            if group["movement"] != "still":  # Don't count still phases in significant movements
                movement = group["movement"]
                significant_movements[movement] = significant_movements.get(movement, 0) + 1
        
        # Count still phases between movements
        still_phases = len([g for g in significant_groups if g["movement"] == "still"])
        
        # Detailed information about each detected segment
        detailed_segments = []
        for group in significant_groups:
            detailed_segments.append({
                "movement": group["movement"],
                "start_time": group["start_time"],
                "end_time": group["end_time"],
                "duration": group["duration"],
                "window_count": group["count"]
            })
        
        return {
            "all_detected_movements": movement_counts,
            "significant_movements": significant_movements,
            "still_phases": still_phases,
            "detailed_segments": detailed_segments,
            "raw_window_predictions": {
                "predictions": window_preds,
                "smoothed_predictions": smoothed_preds,
                "confidences": window_confs,
                "times": window_times
            },
            "window_params": {
                "window_size_ms": window_size_ms,
                "overlap_ms": overlap_ms,
                "sample_rate_hz": sample_rate_hz
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV data: {str(e)}")

@router.post("/predict-window")
async def predict_single_window(csv_data: str = Form(...)):
    """
    TEST ENDPOINT: Predict the atomic movement (tap, wrist_rotation, still) for a single window of data.
    
    This endpoint is for testing the classifier on short examples of atomic movements.
    
    Args:
        csv_data: CSV text data containing IMU readings for a single atomic movement (e.g., 350ms of data)
        
    Returns:
        Classification of the movement and confidence scores
    """
    if not model.trained:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        # Convert string data to bytes and read it
        content = csv_data.encode('utf-8')
        df = read_csv_from_bytes(content)
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Could not parse CSV data. Make sure it's valid CSV format.")
        
        is_valid, error_message = validate_imu_data(df)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Make prediction for this single window
        result = model.predict(df)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))
        
        return {
            "predicted_movement": result["predicted_movement"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV data: {str(e)}")

@router.post("/upload-training-data")
async def upload_training_data(csv_data: str = Form(...), movement_type: Optional[str] = Form(None)):
    """
    Upload CSV data as text for training on atomic movements.
    
    Args:
        csv_data: CSV text data containing IMU readings for a single atomic movement
        movement_type: Type of movement (e.g., 'tap', 'wrist_rotation', 'still'). If not provided,
                      the movement type will be extracted from the recording_id field in the CSV.
    
    Returns:
        Confirmation message with the movement type that was saved
    """
    # Ensure training data directory exists
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    try:
        # Convert string data to bytes and read it
        print(f"Received data length: {len(csv_data)} characters")
        content = csv_data.encode('utf-8')
        
        # Try to read as CSV data
        df = pd.read_csv(io.StringIO(csv_data))
        
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Could not parse CSV data. Make sure it's valid CSV format.")
        
        # Check if required columns are present
        required_columns = ['rel_timestamp', 'recording_id', 
                           'acc_x', 'acc_y', 'acc_z', 
                           'gyro_x', 'gyro_y', 'gyro_z']
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, 
                               detail=f"Missing required columns: {missing}. Found columns: {df.columns.tolist()}")
        
        # Determine movement type - either from parameter or from recording_id
        if movement_type:
            # Use provided movement_type
            gesture_name = movement_type.lower().strip()
        else:
            # Extract recording_id to use as movement name
            if 'recording_id' not in df.columns:
                raise HTTPException(status_code=400, detail="CSV must contain a 'recording_id' column or provide movement_type")
            
            # Use the first recording_id value as the movement name (remove any g_ prefix if present)
            recording_id = str(df['recording_id'].iloc[0])
            gesture_name = recording_id.replace('g_', '') if recording_id.startswith('g_') else recording_id
        
        # Validate movement type (safeguard against unexpected values)
        valid_movement_types = ['tap', 'wrist_rotation', 'still']
        if gesture_name not in valid_movement_types and not gesture_name.startswith(('tap_', 'wrist_rotation_', 'still_')):
            print(f"Warning: Unusual movement type '{gesture_name}'. Expected one of {valid_movement_types}.")
            # We don't raise an exception here, just log a warning
        
        # Save the file
        # Use a naming pattern that includes sample count for organization
        existing_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.startswith(f"{gesture_name}_")]
        sample_num = len(existing_files) + 1
        file_path = os.path.join(TRAINING_DATA_DIR, f"{gesture_name}_{sample_num:03d}.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the file
        df.to_csv(file_path, index=False)
        
        return {"message": f"Training data for movement '{gesture_name}' (sample #{sample_num}) uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV data: {str(e)}")

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

@router.delete("/training-data/{movement_name}")
async def delete_training_data(movement_name: str):
    """
    Delete the training data for a specific atomic movement type.
    """
    # Find all files for this movement type
    files_to_delete = [f for f in os.listdir(TRAINING_DATA_DIR) 
                      if f.startswith(f"{movement_name}_") or f == f"{movement_name}.csv"]
    
    if not files_to_delete:
        raise HTTPException(status_code=404, detail=f"Training data for movement '{movement_name}' not found")
    
    try:
        deleted_count = 0
        for file_name in files_to_delete:
            file_path = os.path.join(TRAINING_DATA_DIR, file_name)
            os.remove(file_path)
            deleted_count += 1
            
        return {"message": f"Deleted {deleted_count} training samples for movement '{movement_name}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files: {str(e)}")

@router.get("/training-data")
async def list_training_data():
    """
    List all available training data for atomic movements.
    """
    if not os.path.exists(TRAINING_DATA_DIR):
        return {"movements": [], "sample_counts": {}}
    
    try:
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.csv')]
        
        # Extract movement types and count samples per movement
        movement_samples = {}
        
        for filename in csv_files:
            # Extract movement name from filename pattern (e.g., "tap_001.csv" -> "tap")
            parts = os.path.splitext(filename)[0].split('_')
            if len(parts) >= 2:
                # Handle pattern like "tap_001" or "wrist_rotation_001"
                if parts[0] == "wrist" and len(parts) >= 3:
                    movement = f"{parts[0]}_{parts[1]}"
                else:
                    movement = parts[0]
            else:
                # Fallback for old naming
                movement = parts[0]
                
            movement_samples[movement] = movement_samples.get(movement, 0) + 1
        
        return {
            "movements": list(movement_samples.keys()),
            "sample_counts": movement_samples,
            "total_samples": len(csv_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing training data: {str(e)}")

@router.delete("/delete-all-training-data")
async def delete_all_training_data():
    """
    Delete all training data files.
    """
    if not os.path.exists(TRAINING_DATA_DIR):
        return {"message": "No training data exists"}
    
    try:
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            return {"message": "No training data files found"}
        
        # Delete each file
        deleted_count = 0
        for file_name in csv_files:
            file_path = os.path.join(TRAINING_DATA_DIR, file_name)
            os.remove(file_path)
            deleted_count += 1
        
        return {"message": f"Successfully deleted {deleted_count} training data files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting training data: {str(e)}") 