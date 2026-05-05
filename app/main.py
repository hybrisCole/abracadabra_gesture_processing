import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router

os.makedirs("app/data/training", exist_ok=True)
os.makedirs("app/data/pending_training", exist_ok=True)

app = FastAPI(
    title=os.getenv("API_TITLE", "Abracadabra Gesture Recognition API"),
    description="JSON-only gesture recognition API for Abracadabra RN recordings",
    version=os.getenv("API_VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to known app origins if you later add browser clients.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "Abracadabra Gesture Recognition API",
        "description": "Train and detect timed gesture-password sequences from RN IMU JSON",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "development"),
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_status": "/api/model-status",
            "model_details": "/api/model-details",
            "train": "/api/train",
            "training_samples": "/api/training-samples",
            "classify_crop": "/api/recordings/classify",
            "analyze_recording": "/api/recordings/analyze",
            "verify_gesture_password": "/api/gesture-passwords/verify",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "gesture-recognition-api",
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "development"),
        "port": os.getenv("PORT", "8000"),
    }
