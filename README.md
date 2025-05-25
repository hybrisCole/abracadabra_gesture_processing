# Gesture Recognition API

A FastAPI-based HTTP API for training and recognizing gestures from IMU sensor data collected from a XIAO nRF52840 Sense microcontroller.

## Features

- Train models to recognize atomic movements (taps, wrist rotations, still periods)
- Upload CSV files containing sensor data for different movement types
- Detect and count occurrences of movements within longer data streams
- Advanced feature extraction for IMU data analysis
- Random Forest classifier for robust movement detection
- RESTful API with Swagger documentation
- Web interface for data collection and testing

## Project Structure

```
.
├── app/                    # Main application package
│   ├── api/                # API routes and endpoints
│   ├── data/               # Directory for storing data and models
│   │   └── training/       # Training data storage
│   ├── models/             # ML models implementation
│   └── utils/              # Utility functions
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd abracadabra_gesture_processing
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the API Server

Run the FastAPI server:

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000.
API documentation is available at http://localhost:8000/docs.

### Web Interface

The system provides several web interfaces:

- **Home**: http://localhost:8000/
- **Atomic Movement Upload Form**: http://localhost:8000/atomic-movement-form
  - Use this specialized form to upload and label short segments of tap, wrist rotation, and still movements
- **Standard Upload Form**: http://localhost:8000/upload-form
- **Model Details**: http://localhost:8000/model-details

### API Endpoints

#### Training Endpoints
- **POST /api/upload-training-data**: Upload CSV training data for a specific movement
- **POST /api/train**: Train the model using uploaded training data
- **GET /api/model-status**: Check the current status of the model
- **GET /api/training-data**: List available training data
- **DELETE /api/training-data/{movement_name}**: Delete training data for a specific movement

#### Prediction Endpoints
- **POST /api/predict**: Submit new 5-second IMU data to find all movements within the stream
- **POST /api/predict-window**: Test classification of a single short movement sample

### CSV Data Format

The CSV files should have the following columns:
- rel_timestamp (relative timestamp)
- recording_id (identifier for the recording)
- acc_x (accelerometer X-axis)
- acc_y (accelerometer Y-axis)
- acc_z (accelerometer Z-axis)
- gyro_x (gyroscope X-axis)
- gyro_y (gyroscope Y-axis)
- gyro_z (gyroscope Z-axis)

## Workflow for Atomic Movement Detection

1. **Collect Training Samples**:
   - Create short CSV samples (~350ms) for "tap" movements
   - Create short CSV samples (~500ms) for "wrist_rotation" movements
   - Create short CSV samples (~300-500ms) for "still" (non-movement) periods
   - Use the Atomic Movement Upload Form to label and upload these samples

2. **Train the Model**:
   - After uploading at least 20-30 samples of each movement type
   - Hit the train endpoint: `curl -X POST http://localhost:8000/api/train`
   - Check training progress via the Model Details page

3. **Test the Model**:
   - Test individual windows with the "Test Single Movement Window" section of the form
   - Validate model accuracy for each movement type

4. **Use for Movement Detection**:
   - Send 5-second CSV data to the `/api/predict` endpoint
   - The API will return counts and timings of each detected movement
   - It will also identify "still" phases between movements

## How It Works

1. The system collects short samples of atomic movements (tap, wrist_rotation, still)
2. A Random Forest model is trained to classify these atomic movements based on extracted features
3. For detection in longer streams, a sliding window approach is used:
   - The 5-second stream is divided into overlapping windows (default 350ms with 250ms overlap)
   - Each window is classified as tap, wrist_rotation, or still
   - Consecutive identical predictions are grouped into movement events
   - The system counts and times these events, returning detailed information 