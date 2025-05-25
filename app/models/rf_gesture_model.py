import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from scipy import signal
import joblib
import os
import copy
import random


class RandomForestGestureModel:
    """
    A gesture recognition model that uses a Random Forest classifier to recognize IMU-based gestures.
    
    This model extracts time and frequency domain features from IMU data and uses them to train
    a Random Forest classifier. It is designed to classify short, atomic movements like "tap", 
    "wrist_rotation", or "still", which can then be detected within longer data streams using
    a sliding window approach.
    """
    
    def __init__(self):
        """Initialize the Random Forest-based gesture recognition model."""
        self.model = RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            max_depth=None,    # Maximum depth of the trees (None means unlimited)
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,   # For reproducibility
            n_jobs=-1,         # Use all available cores
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.gesture_labels = []  # Store unique gesture labels
        self.feature_names = []   # Store feature names for interpretation
        self.feature_importances = {}  # Store feature importance values
        self.cross_val_scores = None  # Store cross-validation scores
        self.conf_matrix = None   # Store confusion matrix
        
        # Store optimal window parameters
        self.optimal_window_size = None  # Can be set after experimentation
        self.optimal_window_overlap = None  # Can be set after experimentation

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess IMU data for feature extraction.
        
        Args:
            data: DataFrame with IMU sensor data
            
        Returns:
            Preprocessed DataFrame with filtered IMU data
        """
        # Select only the IMU columns
        imu_data = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].copy()
        
        # Calculate magnitudes
        imu_data['acc_mag'] = np.sqrt(imu_data['acc_x']**2 + imu_data['acc_y']**2 + imu_data['acc_z']**2)
        imu_data['gyro_mag'] = np.sqrt(imu_data['gyro_x']**2 + imu_data['gyro_y']**2 + imu_data['gyro_z']**2)
        
        # Apply a Butterworth bandpass filter to reduce noise and focus on relevant frequencies
        # Human gestures are typically in the 0.5-20 Hz range
        nyquist = 0.5 * 250  # Assume 250 Hz sampling rate
        low = 0.5 / nyquist
        high = 20.0 / nyquist
        
        # Apply filter if we have enough data points
        # Adjusted minimum threshold to accommodate shorter windows (e.g., 350ms at 250Hz ≈ 88 samples)
        if len(imu_data) > 5:  # Reduced from 10 to 5 for shorter windows
            b, a = signal.butter(3, [low, high], btype='band')
            
            # Apply the filter to all columns
            for col in imu_data.columns:
                try:
                    imu_data[col] = signal.filtfilt(b, a, imu_data[col])
                except Exception as e:
                    # Fall back to a simple moving average if filtering fails
                    # Shorter window size for shorter data segments
                    window_size = min(5, max(3, len(imu_data) // 10))
                    imu_data[col] = imu_data[col].rolling(window=window_size, center=True).mean()
                    imu_data[col] = imu_data[col].fillna(method='bfill').fillna(method='ffill')
        
        return imu_data
    
    def extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract comprehensive time and frequency domain features from IMU data.
        Optimized to work with shorter atomic movement windows.
        
        Args:
            data: DataFrame with preprocessed IMU data
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        columns = data.columns
        
        # Time-domain features
        for col in columns:
            # Basic statistical features
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_max'] = data[col].max()
            features[f'{col}_min'] = data[col].min()
            features[f'{col}_range'] = data[col].max() - data[col].min()
            features[f'{col}_median'] = data[col].median()
            features[f'{col}_mad'] = (data[col] - data[col].mean()).abs().mean()  # Mean absolute deviation
            
            # For short windows, we'll still compute IQR but be aware it might be less meaningful
            features[f'{col}_iqr'] = data[col].quantile(0.75) - data[col].quantile(0.25)  # Interquartile range
            
            # Signal characteristics
            features[f'{col}_rms'] = np.sqrt(np.mean(data[col]**2))  # Root mean square
            
            # Shape features - if we have enough data points
            if len(data) >= 5:
                features[f'{col}_skewness'] = data[col].skew()  # Asymmetry
                features[f'{col}_kurtosis'] = data[col].kurtosis()  # Peakedness
            else:
                features[f'{col}_skewness'] = 0
                features[f'{col}_kurtosis'] = 0
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(data[col].values)))
            features[f'{col}_zero_crossings'] = zero_crossings / len(data)
            
            # Peak detection
            # Find peaks with prominence at least 25% of range
            prominence = 0.25 * features[f'{col}_range']
            if prominence > 0:
                peaks, _ = signal.find_peaks(data[col].values, prominence=prominence)
                features[f'{col}_num_peaks'] = len(peaks)
                # Mean time between peaks in normalized time units
                if len(peaks) > 1:
                    features[f'{col}_mean_peak_distance'] = np.mean(np.diff(peaks)) / len(data)
                else:
                    features[f'{col}_mean_peak_distance'] = 0
            else:
                features[f'{col}_num_peaks'] = 0
                features[f'{col}_mean_peak_distance'] = 0
        
        # Frequency-domain features - adapted for shorter windows
        for col in columns:
            try:
                # Compute the power spectral density - adjust nperseg for shorter windows
                # For short windows (e.g., 350ms at 250Hz ≈ 88 samples), use a smaller nperseg
                nperseg = min(128, max(len(data) // 2, 16))  # At least 16 samples, at most 128
                f, Pxx = signal.welch(data[col].values, fs=250, nperseg=nperseg)
                
                # Dominant frequency components
                if len(Pxx) > 0:
                    features[f'{col}_dom_freq'] = f[np.argmax(Pxx)]
                    features[f'{col}_freq_mean'] = np.average(f, weights=Pxx)
                    features[f'{col}_freq_std'] = np.sqrt(np.average((f - features[f'{col}_freq_mean'])**2, weights=Pxx))
                    
                    # Spectral energy in different frequency bands
                    # Low: 0.5-3 Hz, Mid: 3-10 Hz, High: 10-20 Hz
                    low_idx = np.logical_and(f >= 0.5, f < 3)
                    mid_idx = np.logical_and(f >= 3, f < 10)
                    high_idx = np.logical_and(f >= 10, f < 20)
                    
                    total_energy = np.sum(Pxx)
                    if total_energy > 0:
                        features[f'{col}_low_energy_ratio'] = np.sum(Pxx[low_idx]) / total_energy
                        features[f'{col}_mid_energy_ratio'] = np.sum(Pxx[mid_idx]) / total_energy
                        features[f'{col}_high_energy_ratio'] = np.sum(Pxx[high_idx]) / total_energy
                    else:
                        features[f'{col}_low_energy_ratio'] = 0
                        features[f'{col}_mid_energy_ratio'] = 0
                        features[f'{col}_high_energy_ratio'] = 0
                    
                    # Spectral entropy (complexity of the frequency distribution)
                    if np.any(Pxx > 0):
                        features[f'{col}_spectral_entropy'] = entropy(Pxx / np.sum(Pxx))
                    else:
                        features[f'{col}_spectral_entropy'] = 0
                else:
                    # Default values if PSD computation fails
                    features[f'{col}_dom_freq'] = 0
                    features[f'{col}_freq_mean'] = 0
                    features[f'{col}_freq_std'] = 0
                    features[f'{col}_low_energy_ratio'] = 0
                    features[f'{col}_mid_energy_ratio'] = 0
                    features[f'{col}_high_energy_ratio'] = 0
                    features[f'{col}_spectral_entropy'] = 0
            except Exception as e:
                # Handle cases where frequency analysis fails
                print(f"Frequency analysis failed for {col}: {e}")
                features[f'{col}_dom_freq'] = 0
                features[f'{col}_freq_mean'] = 0
                features[f'{col}_freq_std'] = 0
                features[f'{col}_low_energy_ratio'] = 0
                features[f'{col}_mid_energy_ratio'] = 0
                features[f'{col}_high_energy_ratio'] = 0
                features[f'{col}_spectral_entropy'] = 0
        
        # Correlation features between axes
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Only calculate correlations once per pair
                    corr = data[col1].corr(data[col2])
                    features[f'corr_{col1}_{col2}'] = corr if not np.isnan(corr) else 0
        
        return features
    
    def augment_training_data(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Augment training data by creating variations of each atomic movement sample.
        
        Args:
            training_data: Dictionary mapping atomic movement names to DataFrames of IMU data
            
        Returns:
            Augmented training data dictionary
        """
        augmented_data = copy.deepcopy(training_data)
        
        for gesture_name, original_df in training_data.items():
            # Create 5 variations for each gesture
            for i in range(1, 6):
                # Create a copy of the original data
                augmented_df = original_df.copy()
                
                # Apply different augmentation techniques - modified for shorter segments
                if i == 1:
                    # Time scaling (slightly faster) - adapted for shorter windows
                    # For very short windows, we'll be more conservative with time scaling
                    if len(original_df) > 30:
                        augmented_df = augmented_df.iloc[::2].reset_index(drop=True)
                        # Interpolate back to original length
                        augmented_df = augmented_df.reindex(range(len(original_df))).interpolate()
                    else:
                        # Apply milder time scaling for very short windows
                        sample_step = max(1, int(0.8 * len(original_df)) // len(original_df))
                        if sample_step > 1:
                            augmented_df = augmented_df.iloc[::sample_step].reset_index(drop=True)
                            augmented_df = augmented_df.reindex(range(len(original_df))).interpolate()
                
                elif i == 2:
                    # Time scaling (slightly slower)
                    # For short windows, use milder stretching
                    if len(original_df) > 20:
                        # Double the data and then sample to get original length
                        repeated = pd.concat([augmented_df, augmented_df]).reset_index(drop=True)
                        augmented_df = repeated.iloc[::2][:len(original_df)].reset_index(drop=True)
                
                elif i == 3:
                    # Add small random noise (±5% of original signal)
                    # For short windows, we use slightly less noise to preserve characteristic patterns
                    noise_factor = 0.03 if len(original_df) < 30 else 0.05
                    for col in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
                        if col in augmented_df.columns:
                            noise = np.random.normal(0, noise_factor * augmented_df[col].std(), len(augmented_df))
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
    
    def prepare_training_data(self, training_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data for the Random Forest classifier.
        
        Args:
            training_data: Dictionary mapping atomic movement names to DataFrames of IMU data
            
        Returns:
            Tuple of (X, y, feature_names) where X is the feature matrix,
            y is the target vector, and feature_names is a list of feature names
        """
        X = []  # Feature vectors
        y = []  # Labels
        feature_names = None
        
        for gesture_name, data in training_data.items():
            # Preprocess the data
            preprocessed = self.preprocess_data(data)
            
            # Extract features
            features = self.extract_features(preprocessed)
            
            if feature_names is None:
                feature_names = list(features.keys())
            
            # Convert features to a vector
            feature_vector = [features[name] for name in feature_names]
            
            # Clean any NaN or inf values
            feature_vector = np.array(feature_vector)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Handle the case of augmented data (strip _augX suffix for label)
            if '_aug' in gesture_name:
                base_gesture = gesture_name.split('_aug')[0]
            else:
                # Data is already properly grouped by movement type from data_handler
                base_gesture = gesture_name
            
            # Add to training data
            X.append(feature_vector)
            y.append(base_gesture)
        
        return np.array(X), np.array(y), feature_names
    
    def train(self, training_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train the Random Forest model on atomic movement data.
        
        Args:
            training_data: Dictionary mapping atomic movement names to DataFrames of IMU data
        """
        if not training_data:
            print("No training data provided")
            return
        
        print(f"Training Random Forest model with {len(training_data)} atomic movement examples")
        
        # Augment training data
        augmented_data = self.augment_training_data(training_data)
        print(f"Augmented to {len(augmented_data)} examples")
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_data(augmented_data)
        self.feature_names = feature_names
        
        if len(X) == 0 or len(y) == 0:
            print("No valid training examples extracted")
            return
        
        # Store unique atomic movement labels
        self.gesture_labels = np.unique(y)
        print(f"Detected atomic movements: {', '.join(self.gesture_labels)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation if we have enough samples
        if len(np.unique(y)) > 1 and len(y) >= 5:
            print("Performing cross-validation...")
            skf = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='accuracy')
            self.cross_val_scores = scores
            print(f"Cross-validation scores: {scores}")
            print(f"Mean CV accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
        
        # Train the final model on all data
        self.model.fit(X_scaled, y)
        self.trained = True
        
        # Store feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature ranking:")
        for i, idx in enumerate(indices[:20]):  # Print top 20 features
            self.feature_importances[feature_names[idx]] = importances[idx]
            print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")
    
    def predict(self, query_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict the atomic movement class for a short window of data.
        
        Args:
            query_data: DataFrame with IMU sensor data for a single window
            
        Returns:
            Dictionary with prediction results and probabilities
        """
        if not self.trained:
            return {
                "error": "Model not trained yet",
                "success": False
            }
        
        # Preprocess the query data
        preprocessed_query = self.preprocess_data(query_data)
        
        # Extract features from the query
        query_features = self.extract_features(preprocessed_query)
        
        # Convert features to vector in the same order as training data
        feature_vector = [query_features.get(name, 0) for name in self.feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Clean any NaN or inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(feature_vector)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get class label predictions
        predicted_class = self.model.predict(X_scaled)[0]
        
        # Create a dictionary mapping gesture labels to probabilities
        all_probabilities = {label: prob for label, prob in zip(self.model.classes_, probabilities)}
        
        # Sort by probability (descending)
        sorted_probabilities = dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "predicted_movement": predicted_class,
            "confidence": sorted_probabilities[predicted_class],
            "all_probabilities": sorted_probabilities,
            "success": True
        }

    def predict_window_sequence(self, query_data: pd.DataFrame, window_size_ms: int = 350, 
                              overlap_ms: int = 100, sample_rate_hz: int = 250) -> Dict[str, Any]:
        """
        Predict a sequence of atomic movements within a longer data stream using a sliding window approach.
        This is a helper method to implement the sliding window logic at the model level.
        
        Args:
            query_data: DataFrame with IMU sensor data (longer stream)
            window_size_ms: Window size in milliseconds
            overlap_ms: Overlap between consecutive windows in milliseconds
            sample_rate_hz: Sampling rate in Hz
            
        Returns:
            Dictionary with prediction results including the sequence of detected movements
        """
        if not self.trained:
            return {
                "error": "Model not trained yet",
                "success": False
            }
        
        # Calculate window size and step in samples
        window_samples = int((window_size_ms / 1000) * sample_rate_hz)
        step_samples = int(((window_size_ms - overlap_ms) / 1000) * sample_rate_hz)
        
        # Check if we have enough data
        if len(query_data) < window_samples:
            return {
                "error": f"Data too short for window size of {window_size_ms}ms",
                "success": False
            }
        
        # Sliding window predictions
        window_predictions = []
        window_confidences = []
        window_times = []  # Store the center time of each window for reference
        
        # Process each window
        for start_idx in range(0, len(query_data) - window_samples + 1, step_samples):
            # Extract window data
            end_idx = start_idx + window_samples
            window_data = query_data.iloc[start_idx:end_idx].copy()
            
            # Get window center time (from rel_timestamp)
            if 'rel_timestamp' in window_data.columns:
                center_time = window_data['rel_timestamp'].mean()
            else:
                # If no timestamp, use sample index as an approximation
                center_time = (start_idx + end_idx) / 2 / sample_rate_hz  # in seconds
            
            # Predict for this window
            result = self.predict(window_data)
            
            if result["success"]:
                window_predictions.append(result["predicted_movement"])
                window_confidences.append(result["confidence"])
                window_times.append(center_time)
            else:
                # Handle error if needed
                window_predictions.append("error")
                window_confidences.append(0.0)
                window_times.append(center_time)
        
        return {
            "window_predictions": window_predictions,
            "window_confidences": window_confidences,
            "window_times": window_times,
            "window_size_ms": window_size_ms,
            "overlap_ms": overlap_ms,
            "sample_rate_hz": sample_rate_hz,
            "success": True
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model to
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'trained': self.trained,
            'gesture_labels': self.gesture_labels,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'cross_val_scores': self.cross_val_scores,
            'conf_matrix': self.conf_matrix,
            'optimal_window_size': self.optimal_window_size,
            'optimal_window_overlap': self.optimal_window_overlap
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
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
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.trained = model_data['trained']
            self.gesture_labels = model_data['gesture_labels']
            self.feature_names = model_data['feature_names']
            self.feature_importances = model_data['feature_importances']
            self.cross_val_scores = model_data.get('cross_val_scores', None)
            self.conf_matrix = model_data.get('conf_matrix', None)
            self.optimal_window_size = model_data.get('optimal_window_size', None)
            self.optimal_window_overlap = model_data.get('optimal_window_overlap', None)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False 