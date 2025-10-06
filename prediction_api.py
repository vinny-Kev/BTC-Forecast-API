"""
Prediction API
FastAPI service for serving ML model predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_scraper import DataScraper
from feature_engineering import FeatureEngineer
from models import EnsembleModel

# Initialize FastAPI
app = FastAPI(
    title="Crypto Price Movement Forecasting API",
    description="ML ensemble for predicting large crypto price movements",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
ensemble_model = None
preprocessor = None
feature_engineer = None
feature_columns = None
metadata = None
sequence_length = 30


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    symbol: str = "BTCUSDT"
    interval: str = "30s"
    use_live_data: bool = True


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    symbol: str
    timestamp: str
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    current_price: float
    expected_movement: Optional[float] = None
    next_periods: List[Dict] = []


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


def load_models(model_dir: str):
    """Load trained models and preprocessor"""
    global ensemble_model, preprocessor, feature_engineer, feature_columns, metadata, sequence_length
    
    print(f"Loading models from {model_dir}...")
    
    # Load ensemble
    ensemble_model = EnsembleModel(n_classes=3)
    ensemble_model.load(model_dir)
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    preprocessor = joblib.load(preprocessor_path)
    print(f"✓ Preprocessor loaded")
    
    # Load feature columns
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    feature_columns = joblib.load(feature_cols_path)
    print(f"✓ Feature columns loaded ({len(feature_columns)} features)")
    
    # Load metadata
    import json
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    sequence_length = metadata.get('sequence_length', 30)
    print(f"✓ Metadata loaded (sequence_length: {sequence_length})")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    print(f"✓ Feature engineer initialized")
    
    print("✓ All models loaded successfully!")


def get_latest_model_dir(base_dir='data/models'):
    """Get the most recent model directory"""
    if not os.path.exists(base_dir):
        return None
    
    # Get all subdirectories
    subdirs = [
        os.path.join(base_dir, d) 
        for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    
    if not subdirs:
        return None
    
    # Sort by modification time, get latest
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    # Try to load the latest model (look in parent directory's data/models)
    model_dir = get_latest_model_dir('../data/models')
    
    if model_dir:
        try:
            load_models(model_dir)
            print(f"✓ API ready with models from: {model_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not load models: {e}")
            print("API will run without loaded models. Train models first.")
    else:
        print("⚠ No models found. Please train models first using train_pipeline.py")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=ensemble_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get price movement predictions
    
    Returns:
    - prediction: 0 (no movement), 1 (large up), 2 (large down)
    - confidence: probability of predicted class
    - probabilities: all class probabilities
    """
    
    if ensemble_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        # Fetch live data
        scraper = DataScraper(symbol=request.symbol, interval=request.interval)
        
        # Get enough historical data for feature engineering
        lookback = "6 hours ago UTC"  # Should be enough for all features
        df = scraper.fetch_historical_(lookback)
        
        # Get market context
        _, context = scraper.fetch_context_data()
        
        # Generate features (without targets for prediction)
        df_features = feature_engineer.generate_all_features(
            df,
            order_book_context=context.get('order_book'),
            ticker_context=context.get('ticker'),
            create_targets=False
        )
        
        # Get only the feature columns we trained on
        X = df_features[feature_columns].tail(sequence_length + 1)
        
        # Scale features
        X_scaled = preprocessor.transform(X)
        
        # Prepare for prediction
        # For regular models (CatBoost, RF): use latest data point
        X_regular = X_scaled.tail(1)
        
        # For LSTM: prepare sequence
        X_lstm_seq = X_scaled.tail(sequence_length).values
        X_lstm = np.array([X_lstm_seq])  # Add batch dimension
        
        # Get ensemble predictions
        probabilities = ensemble_model.predict_proba(X_regular, X_lstm)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # Get prediction label
        labels = {
            0: "No Significant Movement",
            1: "Large Upward Movement Expected",
            2: "Large Downward Movement Expected"
        }
        prediction_label = labels[prediction]
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Calculate expected movement magnitude (rough estimate)
        threshold = metadata.get('threshold_percent', 0.5)
        expected_movement = None
        if prediction == 1:
            expected_movement = threshold  # Positive percentage
        elif prediction == 2:
            expected_movement = -threshold  # Negative percentage
        
        # Generate predictions for next periods
        next_periods = []
        for i in range(1, 7):  # Next 6 periods
            future_price = current_price
            if prediction == 1:
                future_price = current_price * (1 + (threshold / 100) * (i / 6))
            elif prediction == 2:
                future_price = current_price * (1 - (threshold / 100) * (i / 6))
            
            next_periods.append({
                'period': i,
                'estimated_price': round(future_price, 2),
                'confidence': round(confidence * (1 - i * 0.05), 3)  # Decay confidence
            })
        
        return PredictionResponse(
            symbol=request.symbol,
            timestamp=datetime.now().isoformat(),
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            probabilities={
                'no_movement': float(probabilities[0]),
                'large_up': float(probabilities[1]),
                'large_down': float(probabilities[2])
            },
            current_price=current_price,
            expected_movement=expected_movement,
            next_periods=next_periods
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about loaded models"""
    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )
    
    return {
        "metadata": metadata,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "sequence_length": sequence_length,
        "model_loaded": ensemble_model is not None
    }


@app.post("/model/reload")
async def reload_models():
    """Reload models from latest directory"""
    model_dir = get_latest_model_dir('../data/models')
    
    if not model_dir:
        raise HTTPException(
            status_code=404,
            detail="No model directory found"
        )
    
    try:
        load_models(model_dir)
        return {
            "status": "success",
            "message": f"Models reloaded from {model_dir}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload models: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting Crypto Price Movement Prediction API")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
