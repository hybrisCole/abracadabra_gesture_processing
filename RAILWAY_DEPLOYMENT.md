# Railway Deployment Guide

## Abracadabra Gesture Recognition API on Railway

This project is configured for Railway as the JSON-only gesture processing server used by `abracadabra-rnapp`.

## Railway Configuration Files

- **`railway.toml`** - Main Railway configuration
- **`Procfile`** - Process definition for Railway
- **`nixpacks.toml`** - Build system configuration
- **`railway.env.example`** - Environment variables template
- **`Dockerfile`** - Container configuration
- **`docker-entrypoint.sh`** - Fixes ownership on **`/app/app/data`** volume mounts, then runs uvicorn as **`appuser`**

## Configuration Details

### railway.toml
- Defines Docker build process
- Sets health check endpoint (`/health`)
- Configures restart policies
- Sets environment variables

### Persistent volume

Mount the Railway volume at:

```text
/app/app/data
```

The service stores JSON training samples and the trained `rf_gesture_model.joblib` under this path.

If you mount a Railway volume at **`/app/app/data`**, the mount is typically **root-owned**. The app runs as **`appuser` (uid 1000)**, which caused **`PermissionError`** on `app/data/training` until fixed.

The **`docker-entrypoint.sh`** starts as **root**, creates **`training`** / **`pending_training`**, runs **`chown -R appuser:appuser /app/app/data`**, then starts **uvicorn** with **`gosu appuser`** so the process stays non-root.

### Environment Variables
Railway automatically provides:
- `PORT` - Dynamic port assignment
- `RAILWAY_ENVIRONMENT` - Deployment environment

Custom variables (set in Railway dashboard):
- `API_TITLE` - API title (optional)
- `API_VERSION` - API version (optional)

## Health Monitoring

The API includes a dedicated health check endpoint:
- **Endpoint**: `/health`
- **Method**: GET
- **Response**: Service status and environment info

## Deployment Process

1. **Automatic**: Push to GitHub triggers Railway deployment
2. **Manual**: Use Railway CLI or dashboard

### Railway CLI Commands
```bash
# Login to Railway
railway login

# Link to existing project
railway link

# Deploy manually
railway up

# View logs
railway logs

# Open in browser
railway open
```

## API Endpoints

- **`/`** - Root endpoint with API information
- **`/health`** - Health check for Railway monitoring
- **`/docs`** - FastAPI documentation
- **`POST /api/training-samples`** - Save one labeled RN crop as JSON
- **`GET /api/training-samples`** - List stored samples
- **`POST /api/train`** - Train Random Forest from JSON samples
- **`GET /api/model-status`** - Model status and labels
- **`POST /api/recordings/classify`** - Classify one cropped window
- **`POST /api/recordings/analyze`** - Detect timed gesture segments in a full 3-4 second recording
- **`POST /api/gesture-passwords/verify`** - Compare detected sequence to expected gesture-password labels

## Monitoring

Railway provides built-in monitoring:
- Health checks every 30 seconds
- Automatic restarts on failure
- Resource usage metrics
- Deployment logs

## Local Development

To run locally with Railway environment:

```bash
# Install dependencies
pip install -e .

# Run with Railway environment variables
railway run uvicorn app.main:app --reload
```

## 📈 Production Features

- ✅ Multi-stage Docker build for optimization
- ✅ Non-root user for security
- ✅ Health checks for reliability
- ✅ Environment-based configuration
- ✅ CORS middleware for web integration
- ✅ Comprehensive API documentation
- ✅ ML model persistence with joblib files

## 🔒 Security

- Container runs as non-root user (`appuser`)
- Environment variables for sensitive configuration
- CORS configured for production use
- Health checks prevent unhealthy deployments

## 📦 Dependencies

Core dependencies managed in `pyproject.toml`:
- FastAPI for API framework
- uvicorn for ASGI server
- scikit-learn for ML models
- pandas for data processing
- numpy for numerical computations

## 🎯 Performance

- Optimized Docker layers for fast builds
- Health check timeout: 300 seconds
- Restart policy: ON_FAILURE with 10 max retries
- Model loading at startup for fast predictions 