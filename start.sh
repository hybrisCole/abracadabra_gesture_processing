#!/bin/bash

# Handle Railway's PORT variable gracefully
if [ -z "$PORT" ] || [ "$PORT" = '$PORT' ] || ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "PORT variable is invalid or empty, using default port 8000"
    PORT=8000
fi

echo "Starting uvicorn on port: $PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port $PORT 