# Railway-optimized single-stage build
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Install Python dependencies directly (no build tools needed)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.108.0 \
    uvicorn==0.27.0 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    dtw-python==1.3.0 \
    python-multipart==0.0.7 \
    pydantic==2.11.3 \
    joblib==1.3.2

# Copy application code
COPY app/ ./app/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Use Railway-standard command
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT 