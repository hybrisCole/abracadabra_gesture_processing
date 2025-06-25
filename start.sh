#!/bin/bash

# Unset any PORT variable that Railway might inject
unset PORT

# Start uvicorn on port 8000
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 