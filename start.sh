#!/bin/bash

# Use Railway's PORT if it's valid, otherwise default to 8000
if [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Using Railway PORT: $PORT"
    exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
else
    echo "PORT not set or invalid, using default port 8000"
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi 