.PHONY: dev install clean restart train status predict stop reset

# Run the development server
dev:
	source venv/bin/activate && uvicorn app.main:app --reload

# Install dependencies
install:
	pip install -r requirements.txt

# Clean up compiled Python files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -name "*.DS_Store" -delete

# Restart the server (kill previous instance if running)
restart:
	pkill -f uvicorn || true
	source venv/bin/activate && uvicorn app.main:app --reload

# Train the model
train:
	curl -X POST http://localhost:8000/api/train

# Check model status
status:
	curl http://localhost:8000/api/model-status

# Make a prediction with a specified CSV file
# Usage: make predict FILE=csv/upuppredict.csv
predict:
	curl -X POST -F "file=@$(FILE)" http://localhost:8000/api/predict

# Stop the server
stop:
	pkill -f uvicorn || true

# Reset - remove model and training data for a fresh start
reset:
	rm -f app/data/gesture_model.joblib
	echo "Model reset complete. You'll need to reload training data." 