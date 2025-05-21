from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import pandas as pd
import io
import os

from app.api.router import router

# Create data directories if they don't exist
os.makedirs("app/data/training", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Gesture Recognition API",
    description="API for training and predicting gestures from IMU sensor data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Gesture Recognition API with Random Forest",
        "docs_url": "/docs",
        "upload_form": "/upload-form",
        "atomic_movement_form": "/atomic-movement-form",
        "model_details": "/model-details",
        "status": "running"
    }

@app.get("/atomic-movement-form", response_class=HTMLResponse)
async def get_atomic_movement_form():
    """
    Return an HTML form specifically for uploading atomic movement training data
    (tap, wrist_rotation, and still periods)
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Atomic Movement Training Data Upload</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; max-width: 1200px; margin: 0 auto; }
            h1, h2, h3 { color: #333; }
            .info-box { background-color: #f0f8ff; padding: 15px; border-left: 4px solid #4169e1; margin-bottom: 20px; }
            form { margin: 20px 0; background-color: #f8f8f8; padding: 20px; border-radius: 5px; }
            textarea { width: 100%; height: 300px; margin-bottom: 10px; font-family: monospace; border: 1px solid #ddd; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; margin-top: 10px; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f8f8; border-left: 4px solid #4CAF50; display: none; }
            .label-control { margin-bottom: 10px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            select { padding: 8px 12px; width: 100%; max-width: 300px; border: 1px solid #ddd; border-radius: 4px; }
            .nav-links { margin: 20px 0; }
            .nav-links a { display: inline-block; background-color: #4169e1; color: white; padding: 8px 15px; 
                           text-decoration: none; margin-right: 10px; border-radius: 3px; }
            .nav-links a:hover { background-color: #3a5bbf; }
            .movement-card { padding: 15px; margin-bottom: 15px; border-radius: 5px; }
            .tap-card { background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50; }
            .wrist-card { background-color: rgba(33, 150, 243, 0.1); border-left: 4px solid #2196F3; }
            .still-card { background-color: rgba(158, 158, 158, 0.1); border-left: 4px solid #9E9E9E; }
            .samples-info { background-color: #fff3cd; padding: 10px; border-radius: 4px; margin-top: 20px; }
            .counter-box { display: flex; justify-content: space-between; max-width: 600px; margin-bottom: 20px; }
            .counter { padding: 10px 15px; border-radius: 5px; text-align: center; flex: 1; margin: 0 5px; }
            .counter-tap { background-color: rgba(76, 175, 80, 0.2); }
            .counter-wrist { background-color: rgba(33, 150, 243, 0.2); }
            .counter-still { background-color: rgba(158, 158, 158, 0.2); }
            .counter-count { font-size: 24px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Atomic Movement Training Data Upload</h1>
        
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/upload-form">Standard Upload Form</a>
            <a href="/model-details">Model Details</a>
            <a href="/api/model-status">API Status</a>
            <a href="/docs">API Documentation</a>
        </div>
        
        <div class="info-box">
            <h3>About Atomic Movement Collection</h3>
            <p>This system is designed to detect individual movements within a longer data stream. For this to work, we need training data for each atomic movement type:</p>
            
            <div class="movement-card tap-card">
                <h4>Tap Movement</h4>
                <p>A quick tap on the sensor or the surface where the sensor is mounted.</p>
                <ul>
                    <li>Duration: ~350ms</li>
                    <li>Characteristics: Sharp acceleration spike followed by damping oscillations</li>
                    <li>Recommended samples: At least 20-30 examples</li>
                </ul>
            </div>
            
            <div class="movement-card wrist-card">
                <h4>Wrist Rotation Movement</h4>
                <p>A rotation of the wrist while wearing the sensor.</p>
                <ul>
                    <li>Duration: ~500ms</li>
                    <li>Characteristics: Smooth gyroscope readings with rotation around primary axis</li>
                    <li>Recommended samples: At least 20-30 examples</li>
                </ul>
            </div>
            
            <div class="movement-card still-card">
                <h4>Still Period</h4>
                <p>Periods of no movement (important for detecting segments between gestures).</p>
                <ul>
                    <li>Duration: 300-500ms</li>
                    <li>Characteristics: Minimal acceleration/gyroscope readings (baseline noise)</li>
                    <li>Recommended samples: At least 20-30 examples</li>
                </ul>
            </div>
            
            <p class="samples-info">After collecting samples of each type, use the <a href="/api/train">/api/train</a> endpoint to train the model.</p>
        </div>
        
        <div id="sampleCounts" class="counter-box">
            <div class="counter counter-tap">
                <div>Tap Samples</div>
                <div id="tapCount" class="counter-count">0</div>
            </div>
            <div class="counter counter-wrist">
                <div>Wrist Rotation Samples</div>
                <div id="wristCount" class="counter-count">0</div>
            </div>
            <div class="counter counter-still">
                <div>Still Samples</div>
                <div id="stillCount" class="counter-count">0</div>
            </div>
        </div>
        
        <form id="uploadForm">
            <h2>Upload Atomic Movement Data</h2>
            
            <div class="label-control">
                <label for="movementType">Movement Type:</label>
                <select id="movementType" name="movementType">
                    <option value="tap">Tap</option>
                    <option value="wrist_rotation">Wrist Rotation</option>
                    <option value="still">Still</option>
                </select>
            </div>
            
            <label for="csvData">Paste your CSV data below:</label>
            <textarea id="csvData" placeholder="Paste your CSV data here (should include rel_timestamp, recording_id, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z columns)..."></textarea>
            
            <button type="submit">Upload Movement Data</button>
            <div id="uploadResult" class="result"></div>
        </form>
        
        <form id="testForm">
            <h2>Test Single Movement Window</h2>
            <p>After training the model, you can test individual movement windows to see if they're correctly classified.</p>
            
            <textarea id="testData" placeholder="Paste a short CSV segment (e.g., 350-500ms) to test the classification..."></textarea>
            
            <button type="submit">Test Classification</button>
            <div id="testResult" class="result"></div>
        </form>
        
        <script>
            // Fetch and display current sample counts when the page loads
            document.addEventListener('DOMContentLoaded', async function() {
                try {
                    const response = await fetch('/api/training-data');
                    if (response.ok) {
                        const data = await response.json();
                        
                        // Update sample counters
                        document.getElementById('tapCount').textContent = data.sample_counts?.tap || 0;
                        document.getElementById('wristCount').textContent = data.sample_counts?.wrist_rotation || 0;
                        document.getElementById('stillCount').textContent = data.sample_counts?.still || 0;
                    }
                } catch (error) {
                    console.error("Error fetching training data counts:", error);
                }
            });
        
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const csvData = document.getElementById('csvData').value;
                const movementType = document.getElementById('movementType').value;
                const result = document.getElementById('uploadResult');
                
                result.style.display = 'block';
                result.innerHTML = 'Processing...';
                
                try {
                    const response = await fetch('/api/upload-training-data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: new URLSearchParams({
                            'csv_data': csvData,
                            'movement_type': movementType
                        })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        result.innerHTML = `<p style="color:green;">✅ ${data.message}</p>`;
                        
                        // Update the counter for this movement type
                        try {
                            const countResponse = await fetch('/api/training-data');
                            if (countResponse.ok) {
                                const countData = await countResponse.json();
                                document.getElementById('tapCount').textContent = countData.sample_counts?.tap || 0;
                                document.getElementById('wristCount').textContent = countData.sample_counts?.wrist_rotation || 0;
                                document.getElementById('stillCount').textContent = countData.sample_counts?.still || 0;
                            }
                        } catch (error) {
                            console.error("Error updating sample counts:", error);
                        }
                    } else {
                        result.innerHTML = `<p style="color:red;">❌ Error: ${data.detail || JSON.stringify(data)}</p>`;
                    }
                } catch (error) {
                    result.innerHTML = `<p style="color:red;">❌ Error: ${error.message}</p>`;
                }
                
                // Clear text area after successful upload
                if (result.innerHTML.includes('✅')) {
                    document.getElementById('csvData').value = '';
                }
            });
            
            document.getElementById('testForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const csvData = document.getElementById('testData').value;
                const result = document.getElementById('testResult');
                
                result.style.display = 'block';
                result.innerHTML = 'Processing...';
                
                try {
                    const response = await fetch('/api/predict-window', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: new URLSearchParams({
                            'csv_data': csvData
                        })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        // Format confidence as percentage
                        const confidence = (data.confidence * 100).toFixed(2);
                        
                        // Basic result information
                        let resultHtml = `
                            <p style="color:green;">✅ Predicted movement: <strong>${data.predicted_movement}</strong></p>
                            <p>Confidence: <strong>${confidence}%</strong></p>
                        `;
                        
                        // Add probability table for all classes
                        if (data.all_probabilities) {
                            resultHtml += `
                                <h3>Probabilities for All Movement Types</h3>
                                <table>
                                    <tr>
                                        <th>Movement</th>
                                        <th>Probability</th>
                                    </tr>
                            `;
                            
                            Object.entries(data.all_probabilities).forEach(([movement, prob]) => {
                                const probability = (prob * 100).toFixed(2);
                                resultHtml += `
                                    <tr>
                                        <td>${movement}</td>
                                        <td>${probability}%</td>
                                    </tr>
                                `;
                            });
                            
                            resultHtml += `</table>`;
                        }
                        
                        result.innerHTML = resultHtml;
                    } else {
                        result.innerHTML = `<p style="color:red;">❌ Error: ${data.detail || JSON.stringify(data)}</p>`;
                    }
                } catch (error) {
                    result.innerHTML = `<p style="color:red;">❌ Error: ${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/upload-form", response_class=HTMLResponse)
async def get_upload_form():
    """
    Return a simple HTML form to upload CSV data
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gesture Recognition with Random Forest</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            h1 { color: #333; }
            .model-info { background-color: #f0f8ff; padding: 15px; border-left: 4px solid #4169e1; margin-bottom: 20px; }
            form { margin: 20px 0; background-color: #f8f8f8; padding: 15px; border-radius: 5px; }
            textarea { width: 100%; height: 300px; margin-bottom: 10px; font-family: monospace; border: 1px solid #ddd; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f8f8; border-left: 4px solid #4CAF50; }
            .feature-container { margin-top: 15px; }
            .feature-bar { background-color: #e0e0e0; height: 20px; margin-bottom: 5px; }
            .feature-fill { background-color: #4CAF50; height: 100%; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .nav-links { margin: 20px 0; }
            .nav-links a { display: inline-block; background-color: #4169e1; color: white; padding: 8px 15px; 
                           text-decoration: none; margin-right: 10px; border-radius: 3px; }
            .nav-links a:hover { background-color: #3a5bbf; }
        </style>
    </head>
    <body>
        <h1>Gesture Recognition with Random Forest</h1>
        
        <div class="nav-links">
            <a href="/model-details">View Model Details</a>
            <a href="/atomic-movement-form">Atomic Movement Upload</a>
            <a href="/api/model-status">API Status</a>
            <a href="/docs">API Documentation</a>
        </div>
        
        <div class="model-info">
            <h3>About the Model</h3>
            <p>This application uses a <strong>Random Forest classifier</strong> to recognize gestures from IMU sensor data. 
            This approach offers significant improvements over the previous DTW-based system:</p>
            <ul>
                <li>Better handling of variations in gesture execution</li>
                <li>Improved performance with limited training data</li>
                <li>Advanced feature extraction from both time and frequency domains</li>
                <li>Resistance to noise in sensor readings</li>
            </ul>
            <p>For best results, record 5-10 examples of each gesture. The model will extract key features and
            learn which aspects of the motion are most distinctive.</p>
        </div>
        
        <form id="trainingForm">
            <h2>Training Data</h2>
            <p>Paste your CSV data below to upload it for training:</p>
            <textarea id="trainingData" placeholder="Paste your CSV data here..."></textarea>
            <button type="submit">Upload Training Data</button>
            <div id="trainingResult" class="result" style="display: none;"></div>
        </form>
        
        <form id="predictionForm">
            <h2>Prediction</h2>
            <p>Paste your CSV data below to recognize the gesture:</p>
            <textarea id="predictionData" placeholder="Paste your CSV data here..."></textarea>
            <button type="submit">Make Prediction</button>
            <div id="predictionResult" class="result" style="display: none;"></div>
            <div id="featureImportance" class="feature-container" style="display: none;">
                <h3>Key Features for This Prediction</h3>
                <div id="featureContainer"></div>
            </div>
        </form>
        
        <script>
            document.getElementById('trainingForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const csvData = document.getElementById('trainingData').value;
                const result = document.getElementById('trainingResult');
                
                result.style.display = 'block';
                result.innerHTML = 'Processing...';
                
                try {
                    const response = await fetch('/api/upload-training-data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: new URLSearchParams({
                            'csv_data': csvData
                        })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        result.innerHTML = `<p style="color:green;">✅ ${data.message}</p>`;
                    } else {
                        result.innerHTML = `<p style="color:red;">❌ Error: ${data.detail || JSON.stringify(data)}</p>`;
                    }
                } catch (error) {
                    result.innerHTML = `<p style="color:red;">❌ Error: ${error.message}</p>`;
                }
            });
            
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const csvData = document.getElementById('predictionData').value;
                const result = document.getElementById('predictionResult');
                const featureImportance = document.getElementById('featureImportance');
                const featureContainer = document.getElementById('featureContainer');
                
                result.style.display = 'block';
                featureImportance.style.display = 'none';
                result.innerHTML = 'Processing...';
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: new URLSearchParams({
                            'csv_data': csvData
                        })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        // Format confidence as percentage
                        const confidence = (data.confidence * 100).toFixed(2);
                        
                        // Basic result information
                        let resultHtml = `
                            <p style="color:green;">✅ Predicted gesture: <strong>${data.predicted_gesture}</strong></p>
                            <p>Confidence: <strong>${confidence}%</strong></p>
                        `;
                        
                        // Add probability table for all classes
                        if (data.all_probabilities) {
                            resultHtml += `
                                <h3>Probabilities for All Gestures</h3>
                                <table>
                                    <tr>
                                        <th>Gesture</th>
                                        <th>Probability</th>
                                    </tr>
                            `;
                            
                            Object.entries(data.all_probabilities).forEach(([gesture, prob]) => {
                                const probability = (prob * 100).toFixed(2);
                                resultHtml += `
                                    <tr>
                                        <td>${gesture}</td>
                                        <td>${probability}%</td>
                                    </tr>
                                `;
                            });
                            
                            resultHtml += `</table>`;
                        }
                        
                        result.innerHTML = resultHtml;
                        
                        // Display feature importance if available
                        if (data.feature_contributions && Object.keys(data.feature_contributions).length > 0) {
                            featureImportance.style.display = 'block';
                            featureContainer.innerHTML = '';
                            
                            // Find max absolute contribution for scaling
                            const contributions = Object.values(data.feature_contributions);
                            const maxContribution = Math.max(...contributions.map(Math.abs));
                            
                            // Create feature bars
                            Object.entries(data.feature_contributions)
                                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                                .forEach(([feature, contribution]) => {
                                    const absContribution = Math.abs(contribution);
                                    const percent = (absContribution / maxContribution * 100).toFixed(1);
                                    const isPositive = contribution >= 0;
                                    
                                    // Create feature bar element
                                    const featureDiv = document.createElement('div');
                                    featureDiv.innerHTML = `
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                            <div>${feature}</div>
                                            <div>${contribution.toFixed(4)}</div>
                                        </div>
                                        <div class="feature-bar">
                                            <div class="feature-fill" style="width: ${percent}%; background-color: ${isPositive ? '#4CAF50' : '#f44336'};"></div>
                                        </div>
                                    `;
                                    featureContainer.appendChild(featureDiv);
                                });
                        }
                    } else {
                        result.innerHTML = `<p style="color:red;">❌ Error: ${data.detail || JSON.stringify(data)}</p>`;
                    }
                } catch (error) {
                    result.innerHTML = `<p style="color:red;">❌ Error: ${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/model-details", response_class=HTMLResponse)
async def get_model_details_page():
    """
    Return a page showing detailed model information
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Random Forest Model Details</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .model-card { background-color: #f8f8f8; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
            .feature-bar { background-color: #e0e0e0; height: 20px; margin-bottom: 5px; }
            .feature-fill { background-color: #4CAF50; height: 100%; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .cv-score { padding: 5px 10px; background-color: #e8f5e9; border-radius: 3px; margin-right: 5px; }
            .button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; text-decoration: none; display: inline-block; }
            .button:hover { background-color: #45a049; }
            .metrics { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
            .metric-card { background-color: #e8f5e9; padding: 15px; border-radius: 5px; flex: 1; min-width: 200px; }
            .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
            .nav-links { margin: 20px 0; }
            .nav-links a { display: inline-block; background-color: #4169e1; color: white; padding: 8px 15px; 
                           text-decoration: none; margin-right: 10px; border-radius: 3px; }
            .nav-links a:hover { background-color: #3a5bbf; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Random Forest Model Details</h1>
            
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/upload-form">Standard Upload Form</a>
                <a href="/atomic-movement-form">Atomic Movement Upload</a>
                <a href="/docs">API Documentation</a>
            </div>
            
            <div class="model-card">
                <h2>Model Information</h2>
                <p>This page shows detailed information about the trained Random Forest model used for gesture recognition.</p>
                
                <div id="loading">Loading model information...</div>
                <div id="modelInfo" style="display: none;">
                    <div class="metrics" id="metrics"></div>
                    
                    <h3>Trained Movements</h3>
                    <div id="gestures"></div>
                    
                    <h3>Cross-Validation Results</h3>
                    <div id="cvResults"></div>
                    
                    <h3>Feature Importance</h3>
                    <p>The chart below shows the top features used by the Random Forest model to make predictions.</p>
                    <div id="featureImportance"></div>
                </div>
                
                <div id="notTrained" style="display: none;">
                    <p>The model has not been trained yet. Upload training data to train the model.</p>
                    <a href="/atomic-movement-form" class="button">Go to Atomic Movement Upload</a>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', async function() {
                const loading = document.getElementById('loading');
                const modelInfo = document.getElementById('modelInfo');
                const notTrained = document.getElementById('notTrained');
                const metrics = document.getElementById('metrics');
                const gestures = document.getElementById('gestures');
                const cvResults = document.getElementById('cvResults');
                const featureImportance = document.getElementById('featureImportance');
                
                try {
                    // Fetch model details
                    const response = await fetch('/api/model-details');
                    
                    if (response.status === 400) {
                        // Model not trained
                        loading.style.display = 'none';
                        notTrained.style.display = 'block';
                        return;
                    }
                    
                    if (!response.ok) {
                        throw new Error('Failed to fetch model details');
                    }
                    
                    const data = await response.json();
                    
                    // Show model info section
                    loading.style.display = 'none';
                    modelInfo.style.display = 'block';
                    
                    // Display metrics
                    metrics.innerHTML = `
                        <div class="metric-card">
                            <div>Movement Types</div>
                            <div class="metric-value">${data.num_movements}</div>
                        </div>
                        <div class="metric-card">
                            <div>Total Features</div>
                            <div class="metric-value">${data.num_features}</div>
                        </div>
                        <div class="metric-card">
                            <div>Model Accuracy</div>
                            <div class="metric-value">${data.cross_validation && data.cross_validation.mean_accuracy ? 
                                (data.cross_validation.mean_accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
                        </div>
                    `;
                    
                    // Display movements (previously gestures)
                    gestures.innerHTML = `
                        <table>
                            <tr>
                                <th>#</th>
                                <th>Movement Type</th>
                            </tr>
                            ${data.movements.map((movement, i) => `
                                <tr>
                                    <td>${i+1}</td>
                                    <td>${movement}</td>
                                </tr>
                            `).join('')}
                        </table>
                    `;
                    
                    // Display cross-validation results
                    if (data.cross_validation && data.cross_validation.scores) {
                        const mean = data.cross_validation.mean_accuracy;
                        const std = data.cross_validation.std_accuracy;
                        cvResults.innerHTML = `
                            <p>Mean Accuracy: <strong>${(mean * 100).toFixed(1)}% ± ${(std * 100).toFixed(1)}%</strong></p>
                            <p>Individual Fold Scores:</p>
                            <div style="display: flex; flex-wrap: wrap;">
                                ${data.cross_validation.scores.map((score, i) => `
                                    <div class="cv-score">Fold ${i+1}: ${(score * 100).toFixed(1)}%</div>
                                `).join('')}
                            </div>
                        `;
                    } else {
                        cvResults.innerHTML = `<p>No cross-validation data available.</p>`;
                    }
                    
                    // Display feature importance
                    if (data.top_features && Object.keys(data.top_features).length > 0) {
                        const features = Object.entries(data.top_features);
                        const maxImportance = Math.max(...features.map(f => f[1]));
                        
                        featureImportance.innerHTML = `
                            ${features.map(([feature, importance]) => {
                                const percent = (importance / maxImportance * 100).toFixed(1);
                                return `
                                    <div style="margin-bottom: 10px;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                            <div>${feature}</div>
                                            <div>${importance.toFixed(4)}</div>
                                        </div>
                                        <div class="feature-bar">
                                            <div class="feature-fill" style="width: ${percent}%;"></div>
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        `;
                    } else {
                        featureImportance.innerHTML = `<p>No feature importance data available.</p>`;
                    }
                    
                } catch (error) {
                    loading.innerHTML = `Error loading model details: ${error.message}`;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/transform-data")
async def transform_data(csv_data: str = Form(...)):
    """
    Transform data from the format with rel_timestamp, recording_id to the format with timeline, accX, etc.
    Takes CSV data as text input rather than a file upload.
    """
    try:
        # Read the CSV from the provided text string
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Check if the file has the expected columns for the new format
        src_columns = ['rel_timestamp', 'recording_id', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        if all(col in df.columns for col in src_columns):
            # Create a new DataFrame with the old column names
            transformed_df = pd.DataFrame({
                'timeline': df['rel_timestamp'],
                'accX': df['acc_x'],
                'accY': df['acc_y'],
                'accZ': df['acc_z'],
                'gyroX': df['gyro_x'],
                'gyroY': df['gyro_y'],
                'gyroZ': df['gyro_z']
            })
            
            # Convert to CSV
            csv_data = transformed_df.to_csv(index=False)
            
            return JSONResponse(
                content={
                    "message": "Data transformed successfully",
                    "data": csv_data
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Input CSV data does not have the expected columns"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error transforming data: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 