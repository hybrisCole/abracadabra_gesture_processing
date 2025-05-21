import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dtw import dtw
import joblib
import os
import copy
import random

class GestureModel:
    def __init__(self):
        """Initialize the gesture recognition model."""
        self.templates = {}  # Dictionary to store template gestures
        self.features = {}   # Dictionary to store extracted features from templates
        self.scaler = StandardScaler()
        self.trained = False
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess IMU data with improved normalization and filtering.
        
        Args:
            data: DataFrame with IMU sensor data
            
        Returns:
            Preprocessed data as numpy array
        """
        # Select only the IMU columns
        imu_data = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].copy()
        
        # Calculate magnitudes (more informative combined channels)
        acc_mag = np.sqrt(imu_data['acc_x']**2 + imu_data['acc_y']**2 + imu_data['acc_z']**2)
        gyro_mag = np.sqrt(imu_data['gyro_x']**2 + imu_data['gyro_y']**2 + imu_data['gyro_z']**2)
        
        # Add magnitudes to the data
        imu_data['acc_mag'] = acc_mag
        imu_data['gyro_mag'] = gyro_mag
        
        # Apply a stronger moving average filter to reduce noise (larger window)
        window_size = 9  # Increased from 5 to 9
        if len(imu_data) >= window_size:
            imu_data = imu_data.rolling(window=window_size, center=True).mean()
            # Fix deprecated fillna with method
            imu_data = imu_data.bfill().ffill()
        
        # Apply more aggressive normalization using MinMaxScaler for each column
        for col in imu_data.columns:
            # Skip if column is all zeros or constant
            if imu_data[col].std() > 1e-6:  # Avoid division by zero or very small values
                imu_data[col] = (imu_data[col] - imu_data[col].mean()) / imu_data[col].std()
        
        return imu_data.values
    
    def augment_training_data(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Augment training data by creating variations of each gesture.
        
        Args:
            training_data: Dictionary mapping gesture names to DataFrames of IMU data
            
        Returns:
            Augmented training data dictionary
        """
        augmented_data = copy.deepcopy(training_data)
        
        for gesture_name, original_df in training_data.items():
            # Create 5 variations for each gesture
            for i in range(1, 6):
                # Create a copy of the original data
                augmented_df = original_df.copy()
                
                # Apply different augmentation techniques
                if i == 1:
                    # Time scaling (slightly faster)
                    augmented_df = augmented_df.iloc[::2].reset_index(drop=True)
                    # Interpolate back to original length
                    augmented_df = augmented_df.reindex(range(len(original_df))).interpolate()
                
                elif i == 2:
                    # Time scaling (slightly slower)
                    # Double the data and then sample to get original length
                    repeated = pd.concat([augmented_df, augmented_df]).reset_index(drop=True)
                    augmented_df = repeated.iloc[::2][:len(original_df)].reset_index(drop=True)
                
                elif i == 3:
                    # Add small random noise (±5% of original signal)
                    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                        if col in augmented_df.columns:
                            noise = np.random.normal(0, 0.05 * augmented_df[col].std(), len(augmented_df))
                            augmented_df[col] = augmented_df[col] + noise
                
                elif i == 4:
                    # Magnitude scaling (slightly stronger gesture)
                    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                        if col in augmented_df.columns:
                            augmented_df[col] = augmented_df[col] * 1.1
                
                elif i == 5:
                    # Magnitude scaling (slightly weaker gesture)
                    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                        if col in augmented_df.columns:
                            augmented_df[col] = augmented_df[col] * 0.9
                
                # Add the augmented data with a new name
                augmented_data[f"{gesture_name}_aug{i}"] = augmented_df
        
        return augmented_data
    
    def extract_features(self, preprocessed_data: np.ndarray) -> Dict[str, float]:
        """
        Extract relevant features from the preprocessed data.
        
        Args:
            preprocessed_data: Numpy array of preprocessed IMU data
            
        Returns:
            Dictionary of extracted features
        """
        # Calculate basic statistical features
        features = {}
        
        # Assuming the last two columns are acc_mag and gyro_mag in preprocessed_data
        column_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_mag', 'gyro_mag']
        
        # Mean and standard deviation for each axis
        for i, axis in enumerate(column_names[:8]):  # Including the magnitude columns
            if i < preprocessed_data.shape[1]:  # Check if column exists
                features[f'{axis}_mean'] = np.mean(preprocessed_data[:, i])
                features[f'{axis}_std'] = np.std(preprocessed_data[:, i])
                features[f'{axis}_max'] = np.max(preprocessed_data[:, i])
                features[f'{axis}_min'] = np.min(preprocessed_data[:, i])
                
                # Additional features for better characterization
                if preprocessed_data.shape[0] > 3:  # Need at least a few points for these
                    # Zero crossing rate (how often the signal changes sign)
                    features[f'{axis}_zero_crossing'] = np.sum(np.diff(np.signbit(preprocessed_data[:, i]))) / len(preprocessed_data)
                    
                    # Energy (sum of squares)
                    features[f'{axis}_energy'] = np.sum(preprocessed_data[:, i]**2) / len(preprocessed_data)
                    
                    # Slope (trend)
                    if len(preprocessed_data) > 1:
                        features[f'{axis}_slope'] = np.polyfit(np.arange(len(preprocessed_data)), preprocessed_data[:, i], 1)[0]
        
        return features
    
    def train(self, training_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train the model using gesture templates.
        
        Args:
            training_data: Dictionary mapping gesture names to DataFrames of IMU data
        """
        # Augment the training data to get more examples
        augmented_data = self.augment_training_data(training_data)
        
        self.templates = {}
        self.features = {}
        
        all_features = []
        
        # Process each gesture template
        for gesture_name, data in augmented_data.items():
            # Preprocess the data
            preprocessed = self.preprocess_data(data)
            
            # Store the template
            self.templates[gesture_name] = preprocessed
            
            # Extract features
            features = self.extract_features(preprocessed)
            self.features[gesture_name] = features
            
            # Collect features for scaling
            all_features.append(list(features.values()))
        
        # Fit the scaler on all features
        if all_features:
            self.scaler.fit(all_features)
            self.trained = True
    
    def compare_with_dtw(self, query_data: np.ndarray, template_data: np.ndarray) -> float:
        """
        Compare query data with a template using Dynamic Time Warping.
        
        Args:
            query_data: Preprocessed IMU data to compare
            template_data: Template IMU data to compare against
            
        Returns:
            DTW distance (lower is more similar)
        """
        # Ensure both sequences have at least 2 points
        if len(query_data) < 2 or len(template_data) < 2:
            return float('inf')  # Return infinity for impossible comparison
        
        # Use DTW to compare the sequences - fixed to handle object return value
        alignment = dtw(query_data, template_data)
        return alignment.distance
    
    def predict(self, query_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the gesture and calculate similarity scores.
        
        Args:
            query_data: DataFrame with IMU sensor data to classify
            
        Returns:
            Dictionary with prediction results and similarity scores
        """
        if not self.trained or not self.templates:
            return {
                "error": "Model not trained yet",
                "success": False
            }
        
        # Preprocess the query data
        preprocessed_query = self.preprocess_data(query_data)
        
        # Extract features from the query
        query_features = self.extract_features(preprocessed_query)
        
        # Calculate similarities using DTW
        similarities = {}
        augmented_similarities = {}
        
        for gesture_name, template in self.templates.items():
            dtw_distance = self.compare_with_dtw(preprocessed_query, template)
            # Convert distance to similarity (1 / (1 + distance))
            similarity = 1 / (1 + dtw_distance)
            
            # Store similarity
            similarities[gesture_name] = similarity
            
            # For reporting, group augmented versions with original gestures
            base_name = gesture_name.split('_aug')[0]
            if base_name not in augmented_similarities:
                augmented_similarities[base_name] = []
            augmented_similarities[base_name].append(similarity)
        
        # Aggregate augmented similarities - take the max similarity for each gesture type
        aggregated_similarities = {}
        for base_name, sims in augmented_similarities.items():
            aggregated_similarities[base_name] = max(sims)
        
        # Find the best match among aggregated similarities
        best_match = max(aggregated_similarities, key=aggregated_similarities.get)
        
        return {
            "predicted_gesture": best_match,
            "confidence": aggregated_similarities[best_match],
            "all_similarities": aggregated_similarities,  # Return only original gesture names
            "detailed_similarities": similarities,  # Include all templates for debugging
            "success": True
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model to
        """
        model_data = {
            'templates': self.templates,
            'features': self.features,
            'scaler': self.scaler,
            'trained': self.trained
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)
            self.templates = model_data['templates']
            self.features = model_data['features']
            self.scaler = model_data['scaler']
            self.trained = model_data['trained']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False 