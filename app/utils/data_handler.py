import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import io

def read_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read IMU data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with IMU data or None if there was an error
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure all required columns are present
        required_columns = ['rel_timestamp', 'recording_id', 
                           'acc_x', 'acc_y', 'acc_z', 
                           'gyro_x', 'gyro_y', 'gyro_z']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing required columns: {missing}")
            return None
        
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def read_csv_from_bytes(content: bytes) -> Optional[pd.DataFrame]:
    """
    Read IMU data from a bytes object.
    
    Args:
        content: Bytes object containing CSV data
        
    Returns:
        DataFrame with IMU data or None if there was an error
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
        
        # Ensure all required columns are present
        required_columns = ['rel_timestamp', 'recording_id', 
                           'acc_x', 'acc_y', 'acc_z', 
                           'gyro_x', 'gyro_y', 'gyro_z']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing required columns: {missing}")
            return None
        
        return df
    except Exception as e:
        print(f"Error reading CSV from bytes: {e}")
        return None

def read_training_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Read training data from a directory containing CSV files and group them by movement type.
    
    Args:
        data_dir: Directory containing CSV files for training
        
    Returns:
        Dictionary mapping base movement types to concatenated DataFrames
    """
    training_data = {}
    
    try:
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        # Group files by movement type
        movement_files = {}
        
        for csv_file in csv_files:
            # Extract base movement type from filename
            # e.g., "tap_001.csv" -> "tap", "still_021.csv" -> "still"
            filename_no_ext = os.path.splitext(csv_file)[0]
            parts = filename_no_ext.split('_')
            
            if len(parts) >= 2 and parts[-1].isdigit():
                # Handle numbered samples like "tap_001", "wrist_rotation_005"
                base_movement = '_'.join(parts[:-1])
            else:
                # Handle simple names like "tap"
                base_movement = filename_no_ext
            
            if base_movement not in movement_files:
                movement_files[base_movement] = []
            movement_files[base_movement].append(csv_file)
        
        # Read and concatenate files for each movement type
        for movement_type, files in movement_files.items():
            dfs = []
            for csv_file in files:
                file_path = os.path.join(data_dir, csv_file)
                df = read_csv_file(file_path)
                if df is not None:
                    dfs.append(df)
            
            if dfs:
                # Concatenate all DataFrames for this movement type
                combined_df = pd.concat(dfs, ignore_index=True)
                training_data[movement_type] = combined_df
                print(f"Loaded {len(dfs)} samples for movement '{movement_type}' with {len(combined_df)} total data points")
    
    except Exception as e:
        print(f"Error reading training data: {e}")
    
    return training_data

def validate_imu_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate IMU data in a DataFrame.
    
    Args:
        df: DataFrame with IMU data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if df is None or df.empty:
        return False, "Data is empty"
    
    # Check if required columns are present
    required_columns = ['rel_timestamp', 'recording_id', 
                       'acc_x', 'acc_y', 'acc_z', 
                       'gyro_x', 'gyro_y', 'gyro_z']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        return False, f"Missing required columns: {missing}"
    
    # Check if there are enough data points (at least 10)
    if len(df) < 10:
        return False, f"Not enough data points: {len(df)} (minimum 10 required)"
    
    # Check for NaN values
    if df[required_columns].isna().any().any():
        return False, "Data contains NaN values"
    
    return True, "" 