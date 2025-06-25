# Railway Deployment Guide

## 🚀 Abracadabra Gesture Recognition API on Railway

This project is configured for deployment on Railway with proper configuration files and best practices.

## 📁 Railway Configuration Files

- **`railway.toml`** - Main Railway configuration
- **`Procfile`** - Process definition for Railway
- **`nixpacks.toml`** - Build system configuration
- **`railway.env.example`** - Environment variables template
- **`Dockerfile`** - Container configuration

## 🔧 Configuration Details

### railway.toml
- Defines Docker build process
- Sets health check endpoint (`/health`)
- Configures restart policies
- Sets environment variables

### Environment Variables
Railway automatically provides:
- `PORT` - Dynamic port assignment
- `RAILWAY_ENVIRONMENT` - Deployment environment

Custom variables (set in Railway dashboard):
- `API_TITLE` - API title (optional)
- `API_VERSION` - API version (optional)

## 🏥 Health Monitoring

The API includes a dedicated health check endpoint:
- **Endpoint**: `/health`
- **Method**: GET
- **Response**: Service status and environment info

## 🚀 Deployment Process

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

## 📊 API Endpoints

- **`/`** - Root endpoint with API information
- **`/health`** - Health check for Railway monitoring
- **`/docs`** - FastAPI documentation
- **`/api/predict`** - Gesture prediction endpoint
- **`/api/train`** - Model training endpoint
- **`/upload-form`** - Web interface for data upload

## 🔍 Monitoring

Railway provides built-in monitoring:
- Health checks every 30 seconds
- Automatic restarts on failure
- Resource usage metrics
- Deployment logs

## 🛠️ Local Development

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