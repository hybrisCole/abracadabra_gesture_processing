# Gesture Recognition API

A FastAPI-based HTTP API for training and recognizing gestures from IMU sensor data collected from a XIAO nRF52840 Sense microcontroller.

## Features

- Train gesture recognition models using IMU data (accelerometer and gyroscope)
- Upload CSV files containing sensor data for different gestures
- Compare new gesture data against trained models to find the closest match
- Dynamic Time Warping (DTW) algorithm for sequence comparison
- RESTful API with Swagger documentation

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

### API Endpoints

- **POST /api/upload-training-data/{gesture_name}**: Upload CSV training data for a specific gesture
- **POST /api/train**: Train the model using uploaded training data
- **POST /api/predict**: Submit new IMU data to find the closest matching gesture
- **GET /api/model-status**: Check the current status of the model
- **GET /api/training-data**: List available training data
- **DELETE /api/training-data/{gesture_name}**: Delete training data for a specific gesture

### CSV Data Format

The CSV files should have the following columns:
- timeline
- accX (accelerometer X-axis)
- accY (accelerometer Y-axis)
- accZ (accelerometer Z-axis)
- gyroX (gyroscope X-axis)
- gyroY (gyroscope Y-axis)
- gyroZ (gyroscope Z-axis)

## Example Workflow

1. Upload training data for different gestures:
   ```
   curl -X POST -F "file=@gesture1.csv" http://localhost:8000/api/upload-training-data/gesture1
   curl -X POST -F "file=@gesture2.csv" http://localhost:8000/api/upload-training-data/gesture2
   curl -X POST -F "file=@gesture3.csv" http://localhost:8000/api/upload-training-data/gesture3
   ```

2. Train the model:
   ```
   curl -X POST http://localhost:8000/api/train
   ```

3. Check model status:
   ```
   curl http://localhost:8000/api/model-status
   ```

4. Predict a gesture from new data:
   ```
   curl -X POST -F "file=@new_gesture.csv" http://localhost:8000/api/predict
   ```

## How It Works

1. The system collects IMU data from a XIAO nRF52840 Sense microcontroller in CSV format.
2. Training data for different gestures is uploaded to the server.
3. The model processes the training data, extracting features and storing templates.
4. For gesture recognition, the system uses Dynamic Time Warping (DTW) to compare new data with stored templates.
5. The API returns the closest matching gesture along with similarity scores. 