#!/bin/bash

# Force port 8000 - ignore Railway's broken PORT variable
echo "Starting uvicorn on port 8000 (ignoring Railway PORT variable)"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 