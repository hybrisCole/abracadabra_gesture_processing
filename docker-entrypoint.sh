#!/bin/sh
set -e
# Railway (and other) volume mounts are often root-owned; app runs as appuser (uid 1000).
DATA_DIR=/app/app/data
mkdir -p "$DATA_DIR/training" "$DATA_DIR/pending_training"
chown -R appuser:appuser "$DATA_DIR"

# docker-compose may replace CMD with a custom command (e.g. --reload).
if [ "$#" -gt 0 ]; then
  exec gosu appuser "$@"
fi

PORT="${PORT:-8000}"
exec gosu appuser uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
