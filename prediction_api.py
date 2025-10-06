"""
Prediction API
FastAPI service for serving ML model predictions
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import List, Dict, Optional, Annotated
from datetime import datetime, timedelta
import secrets
import json
from collections import defaultdict
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules - defer DataScraper initialization
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

# API Key storage (in production, use a database)
API_KEYS_FILE = 'api_keys.json'
api_keys_db = {}
rate_limit_tracker = defaultdict(list)  # {api_key: [timestamp1, timestamp2, ...]}
guest_usage_tracker = defaultdict(int)  # {ip_address: call_count}

# Rate limiting configuration
RATE_LIMIT_CALLS = 3  # calls per window for authenticated users
RATE_LIMIT_WINDOW = 60  # seconds (1 minute)
GUEST_FREE_CALLS = 3  # free calls for guests (lifetime, not per minute)

# Security
security = HTTPBearer(auto_error=False)

def load_api_keys():
    """Load API keys from file"""
    global api_keys_db, guest_usage_tracker
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, 'r') as f:
                data = json.load(f)
                api_keys_db = data.get('api_keys', {})
                guest_usage_tracker = defaultdict(int, data.get('guest_usage', {}))
            print(f"✓ Loaded {len(api_keys_db)} API keys and {len(guest_usage_tracker)} guest records")
        except Exception as e:
            print(f"⚠ Error loading API keys: {e}")
            api_keys_db = {}
            guest_usage_tracker = defaultdict(int)
    else:
        api_keys_db = {}
        guest_usage_tracker = defaultdict(int)
        print("⚠ No API keys file found, starting fresh")

def save_api_keys():
    """Save API keys to file"""
    try:
        data = {
            'api_keys': api_keys_db,
            'guest_usage': dict(guest_usage_tracker)
        }
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(api_keys_db)} API keys and {len(guest_usage_tracker)} guest records")
    except Exception as e:
        print(f"⚠ Error saving API keys: {e}")

def check_rate_limit(api_key: str) -> bool:
    """Check if API key has exceeded rate limit"""
    now = time.time()
    
    # Clean old timestamps
    rate_limit_tracker[api_key] = [
        ts for ts in rate_limit_tracker[api_key]
        if now - ts < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_tracker[api_key]) >= RATE_LIMIT_CALLS:
        return False
    
    # Add current timestamp
    rate_limit_tracker[api_key].append(now)
    return True

def get_rate_limit_reset_time(api_key: str) -> int:
    """Get seconds until rate limit resets"""
    if not rate_limit_tracker[api_key]:
        return 0
    
    oldest_call = min(rate_limit_tracker[api_key])
    reset_time = oldest_call + RATE_LIMIT_WINDOW
    return max(0, int(reset_time - time.time()))

def verify_api_key(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header or allow guest access"""
    client_ip = request.client.host
    
    # Check if API key is provided
    if credentials:
        api_key = credentials.credentials
        
        if api_key not in api_keys_db:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        # Check rate limit for authenticated users
        if not check_rate_limit(api_key):
            reset_time = get_rate_limit_reset_time(api_key)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. You can make {RATE_LIMIT_CALLS} predictions per minute. Try again in {reset_time} seconds.",
                headers={"Retry-After": str(reset_time)}
            )
        
        return {"type": "authenticated", "data": api_keys_db[api_key], "key": api_key}
    
    # Guest user - check if they have free calls remaining
    else:
        if guest_usage_tracker[client_ip] >= GUEST_FREE_CALLS:
            raise HTTPException(
                status_code=403,
                detail=f"Free trial limit reached ({GUEST_FREE_CALLS} predictions). Please contact Kevin Maglaqui (kevinroymaglaqui27@gmail.com) for an API key to continue using the service."
            )
        
        # Increment guest usage
        guest_usage_tracker[client_ip] += 1
        save_api_keys()  # Persist guest usage
        
        remaining = GUEST_FREE_CALLS - guest_usage_tracker[client_ip]
        return {
            "type": "guest",
            "ip": client_ip,
            "calls_used": guest_usage_tracker[client_ip],
            "calls_remaining": remaining
        }


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
    model_config = {"protected_namespaces": ()}  # Allow 'model_' prefix
    
    status: str
    model_loaded: bool
    timestamp: str


def load_models(model_dir: str):
    """Load trained models and preprocessor"""
    global ensemble_model, preprocessor, feature_engineer, feature_columns, metadata, sequence_length
    
    print(f"Loading models from {model_dir}...")
    
    # Check if directory exists and list contents
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Directory contents: {os.listdir(model_dir)}")
    
    # Load ensemble (CatBoost, Random Forest, Logistic Regression)
    ensemble_model = EnsembleModel(n_classes=3)
    ensemble_model.load(model_dir)
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print(f"✓ Preprocessor loaded")
    else:
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    
    # Load feature columns
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    if os.path.exists(feature_cols_path):
        feature_columns = joblib.load(feature_cols_path)
        print(f"✓ Feature columns loaded ({len(feature_columns)} features)")
    else:
        raise FileNotFoundError(f"Feature columns not found: {feature_cols_path}")
    
    # Load metadata
    import json
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        sequence_length = metadata.get('sequence_length', 30)
        print(f"✓ Metadata loaded (sequence_length: {sequence_length})")
    else:
        print(f"⚠ Metadata not found at {metadata_path}, using defaults")
        metadata = {"sequence_length": 30, "threshold_percent": 0.5}
        sequence_length = 30
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    print(f"✓ Feature engineer initialized")
    
    print("✓ Model loading completed!")


def get_latest_model_dir(base_dir='data/models'):
    """Get the most recent model directory"""
    # Try different possible paths
    possible_paths = [
        base_dir,
        './data/models',
        'data/models',
        os.path.join(os.path.dirname(__file__), 'data', 'models')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Get all subdirectories
            subdirs = [
                os.path.join(path, d) 
                for d in os.listdir(path) 
                if os.path.isdir(os.path.join(path, d))
            ]
            
            if subdirs:
                # Sort by modification time, get latest
                latest_dir = max(subdirs, key=os.path.getmtime)
                return latest_dir
    
    return None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global ensemble_model
    
    print(f"Starting up API...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Load API keys
    load_api_keys()
    
    # Try to load the latest model
    model_dir = get_latest_model_dir()
    
    if model_dir:
        print(f"Found model directory: {model_dir}")
        try:
            load_models(model_dir)
            print(f"✓ API ready with models from: {model_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not load models: {e}")
            print("API will run without loaded models. Train models first.")
            ensemble_model = None
    else:
        print("⚠ No models found in any of the expected locations:")
        print("  - ./data/models")
        print("  - data/models") 
        print(f"  - {os.path.join(os.path.dirname(__file__), 'data', 'models')}")
        print("Please train models first using the training pipeline.")
        ensemble_model = None


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=ensemble_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health endpoint for deployment monitoring"""
    return HealthResponse(
        status="healthy",
        model_loaded=ensemble_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    auth_data: dict = Depends(verify_api_key)
):
    """
    Get price movement predictions
    
    Free Trial: 3 predictions per IP address (no API key needed)
    With API Key: 3 predictions per minute
    
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
        # Import DataScraper only when needed (to defer API key validation)
        from data_scraper import DataScraper
        
        # Fetch live data
        scraper = DataScraper(symbol=prediction_request.symbol, interval=prediction_request.interval)
        
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
        
        # Convert back to DataFrame to use .tail() method
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        # Prepare for prediction
        # For 3-model ensemble (CatBoost, RF, Logistic): use latest data point
        X_regular = X_scaled_df.tail(1).values
        
        # Get ensemble predictions (no LSTM sequences needed)
        probabilities = ensemble_model.predict_proba(X_regular)[0]
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
            symbol=prediction_request.symbol,
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
    
    # Extract performance metrics
    performance = metadata.get('performance', {})
    train_perf = performance.get('train', {})
    test_perf = performance.get('test', {})
    
    return {
        "metadata": metadata,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "n_features": metadata.get('n_features', len(feature_columns) if feature_columns else 0),
        "sequence_length": sequence_length,
        "model_loaded": ensemble_model is not None,
        "symbol": metadata.get('symbol', 'N/A'),
        "interval": metadata.get('interval', 'N/A'),
        "training_date": metadata.get('training_date', 'N/A'),
        "model_type": "3-Model Ensemble (CatBoost + Random Forest + Logistic Regression)",
        "performance": {
            "train_accuracy": train_perf.get('accuracy', 0),
            "test_accuracy": test_perf.get('accuracy', 0),
            "train_f1": train_perf.get('f1_macro', 0),
            "test_f1": test_perf.get('f1_macro', 0),
            "train_roc_auc": train_perf.get('roc_auc_ovr', 0),
            "test_roc_auc": test_perf.get('roc_auc_ovr', 0)
        },
        "train_samples": metadata.get('train_samples', 'N/A'),
        "test_samples": metadata.get('test_samples', 'N/A')
    }


@app.post("/model/reload")
async def reload_models():
    """Reload models from latest directory"""
    model_dir = get_latest_model_dir()
    
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


# ========== API Key Management Endpoints ==========

@app.post("/api-keys/generate")
async def generate_api_key(
    name: str,
    email: str,
    admin_secret: str = Header(..., alias="X-Admin-Secret")
):
    """
    Generate a new API key (Admin only)
    
    Requires X-Admin-Secret header for security
    """
    # Simple admin check (in production, use proper auth)
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change_this_secret_key")
    
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Generate unique API key
    api_key = f"btc_{secrets.token_urlsafe(32)}"
    
    # Store API key
    api_keys_db[api_key] = {
        "name": name,
        "email": email,
        "created_at": datetime.now().isoformat(),
        "calls_made": 0
    }
    
    save_api_keys()
    
    return {
        "success": True,
        "api_key": api_key,
        "name": name,
        "email": email,
        "rate_limit": f"{RATE_LIMIT_CALLS} predictions per minute",
        "usage": f"Include header: Authorization: Bearer {api_key}"
    }


@app.get("/api-keys/usage")
async def get_usage(request: Request, auth_data: dict = Depends(verify_api_key)):
    """Get current usage information"""
    
    if auth_data["type"] == "guest":
        return {
            "user_type": "guest",
            "ip_address": auth_data["ip"],
            "calls_used": auth_data["calls_used"],
            "calls_remaining": auth_data["calls_remaining"],
            "message": f"You have {auth_data['calls_remaining']} free predictions remaining. Contact Kevin Maglaqui for an API key."
        }
    else:
        api_key = auth_data["key"]
        key_data = auth_data["data"]
        
        # Count recent calls
        now = time.time()
        recent_calls = [
            ts for ts in rate_limit_tracker[api_key]
            if now - ts < RATE_LIMIT_WINDOW
        ]
        
        return {
            "user_type": "authenticated",
            "name": key_data.get("name", "Unknown"),
            "email": key_data.get("email", "Unknown"),
            "rate_limit": f"{RATE_LIMIT_CALLS} predictions per minute",
            "calls_in_last_minute": len(recent_calls),
            "calls_remaining": max(0, RATE_LIMIT_CALLS - len(recent_calls))
        }


@app.delete("/api-keys/revoke")
async def revoke_api_key(
    api_key_to_revoke: str,
    admin_secret: str = Header(..., alias="X-Admin-Secret")
):
    """Revoke an API key (Admin only)"""
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change_this_secret_key")
    
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if api_key_to_revoke not in api_keys_db:
        raise HTTPException(status_code=404, detail="API key not found")
    
    del api_keys_db[api_key_to_revoke]
    save_api_keys()
    
    return {
        "success": True,
        "message": "API key revoked successfully"
    }


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
