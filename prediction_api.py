"""
Prediction API
FastAPI service for serving ML model predictions
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
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
import pickle

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from feature_engineering import FeatureEngineer
from models import EnsembleModel
from database import db  # MongoDB database instance

# Initialize FastAPI
app = FastAPI(
    title="Crypto Price Movement Forecasting API",
    description="ML ensemble for predicting large crypto price movements with MongoDB backend",
    version="1.1.0"
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

# Security
security = HTTPBearer(auto_error=False)

# Logger
import logging
logger = logging.getLogger(__name__)

async def verify_api_key(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify API key from Authorization header or allow guest access
    Uses MongoDB for persistent storage
    FAILS SECURE: Rejects all requests if database is unavailable
    """
    client_ip = request.client.host
    
    # Check if MongoDB is connected - FAIL SECURE
    if not db.client:
        logger.error("MongoDB unavailable - rejecting request for security")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Database connection required for authentication."
        )
    
    try:
        # Check if API key is provided
        if credentials:
            api_key = credentials.credentials
            
            # Verify API key with database
            user = await db.get_user_by_api_key(api_key)
            
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key"
                )
            
            # Check and increment user's quota
            can_proceed = await db.increment_user_requests(user["_id"])
            
            if not can_proceed:
                usage = await db.get_user_usage(user["_id"])
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily quota exceeded. Limit: {usage['quota_limit']} requests per day. Resets daily.",
                    headers={"Retry-After": "86400"}  # 24 hours
                )
            
            return {
                "type": "authenticated",
                "user_id": user["_id"],
                "email": user["email"],
                "name": user["name"]
            }
        
        # Guest access (3 free calls total, persisted in database)
        guest_usage = await db.get_guest_usage(client_ip)
        
        if guest_usage["calls_used"] >= guest_usage["calls_limit"]:
            raise HTTPException(
                status_code=403,
                detail=f"Free trial expired. You've used all {guest_usage['calls_limit']} free predictions. Contact Kevin Maglaqui (kevinroymaglaqui27@gmail.com) for an API key."
            )
        
        # Increment guest usage
        await db.increment_guest_usage(client_ip)
        
        return {
            "type": "guest",
            "ip": client_ip,
            "calls_used": guest_usage["calls_used"] + 1,
            "calls_remaining": guest_usage["calls_limit"] - guest_usage["calls_used"] - 1
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions (auth failures, quota exceeded)
        raise
    except Exception as e:
        # Database error during request - FAIL SECURE
        logger.error(f"Database error in verify_api_key: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Database error during authentication."
        )


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


class TrendInfo(BaseModel):
    """Trend direction and strength"""
    short_term: str = Field(..., description="Short-term trend: bullish/bearish/neutral")
    long_term: str = Field(..., description="Long-term trend: bullish/bearish/neutral")
    strength: float = Field(..., description="Trend strength from 0.0 to 1.0", ge=0.0, le=1.0)


class ScoreBreakdown(BaseModel):
    """Score breakdown for suggestion"""
    confidence_boost: int
    trend_score: int
    total_score: int


class TradingSuggestion(BaseModel):
    """Actionable trading suggestion"""
    action: str = Field(..., description="Suggested action: BUY/SELL/WAIT/HOLD")
    conviction: str = Field(..., description="Conviction level: low/medium/high")
    reasoning: List[str] = Field(..., description="List of reasons for suggestion")
    risk_level: str = Field(..., description="Risk level: low/medium/high")
    score_breakdown: ScoreBreakdown


class PredictionResponseV1_1(BaseModel):
    """Enhanced prediction response V1.1 with enriched analysis"""
    # Core prediction
    symbol: str
    timestamp: str
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    current_price: float
    
    # V1.1 Enhancements
    trend: TrendInfo
    tags: List[str] = Field(default_factory=list, description="Contextual market tags")
    suggestion: TradingSuggestion
    
    # Metadata
    api_version: str = "1.1"
    model_version: str
    feature_count: int


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
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        sequence_length = metadata.get('sequence_length', 30)
        feature_columns = metadata.get('feature_columns', [])
        preprocessor_path = metadata.get('preprocessor', 'preprocessor.pkl')
        model_type = metadata.get('model_type', 'unknown')
        print(f"\u2713 Metadata loaded (sequence_length: {sequence_length}, model_type: {model_type})")
    else:
        print(f"\u26a0 Metadata not found at {metadata_path}, using defaults")
        metadata = {"sequence_length": 30, "threshold_percent": 0.5, "feature_columns": [], "model_type": "unknown"}
        sequence_length = 30
        feature_columns = []
        preprocessor_path = 'preprocessor.pkl'
        model_type = 'unknown'

    # Load preprocessor
    if os.path.exists(preprocessor_path):
        try:
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print(f"\u2713 Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            print(f"\u26a0 Could not load preprocessor: {e}")
            raise
    else:
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

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
    """Load models and connect to MongoDB on startup"""
    global ensemble_model
    
    print(f"Starting up API...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Connect to MongoDB - REQUIRED for security
    try:
        await db.connect()
        print("✓ MongoDB connection established")
    except Exception as e:
        print(f"⚠ MongoDB connection failed: {e}")
        print("⚠ WARNING: API will reject all requests until database is connected")
        print("⚠ This is intentional for security - authentication requires database")
    
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


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MongoDB on shutdown"""
    await db.disconnect()
    print("✓ Disconnected from MongoDB")


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


@app.get("/database/health")
async def database_health_check():
    """
    Database health check endpoint - validates MongoDB connection with SSL handshake
    Returns detailed connection status for debugging
    """
    try:
        # Check if MongoDB client exists
        if not db.client:
            return {
                "status": "disconnected",
                "connected": False,
                "ssl_handshake": "not_attempted",
                "error": "MongoDB client not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        # Attempt to ping the database (this performs SSL handshake)
        start_time = datetime.now()
        await db.client.admin.command('ping')
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Get server info for validation
        server_info = await db.client.server_info()
        
        return {
            "status": "connected",
            "connected": True,
            "ssl_handshake": "successful",
            "response_time_seconds": round(response_time, 3),
            "mongodb_version": server_info.get("version"),
            "database_name": db.db.name if db.db is not None else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "ssl_handshake": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }
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
        # Fetch live data
        from data_scraper import DataScraper
        scraper = DataScraper(symbol=prediction_request.symbol, interval=prediction_request.interval)
        df = scraper.fetch_historical_("6 hours ago UTC")
        _, context = scraper.fetch_context_data()

        # Generate features
        df_features = feature_engineer.generate_all_features(
            df,
            order_book_context=context.get('order_book'),
            ticker_context=context.get('ticker'),
            create_targets=False
        )

        # Ensure feature columns match
        X = df_features[feature_columns].tail(sequence_length + 1)
        X_scaled = preprocessor.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)

        # Prepare for prediction
        X_regular = X_scaled_df.tail(1).values
        probabilities = ensemble_model.predict_proba(X_regular)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        labels = {
            0: "No Significant Movement",
            1: "Large Upward Movement Expected",
            2: "Large Downward Movement Expected"
        }
        prediction_label = labels[prediction]
        current_price = float(df['close'].iloc[-1])
        threshold = metadata.get('threshold_percent', 0.5)
        expected_movement = threshold if prediction == 1 else -threshold if prediction == 2 else None

        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            probabilities=probabilities.tolist(),
            current_price=current_price,
            expected_movement=expected_movement
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/v1.1/predict", response_model=PredictionResponseV1_1)
async def predict_v1_1(
    request: Request,
    prediction_request: PredictionRequest,
    auth_data: dict = Depends(verify_api_key)
):
    """
    Enhanced prediction endpoint V1.1 with trend analysis, contextual tags, and trading suggestions
    
    Free Trial: 3 predictions per IP address (no API key needed)
    With API Key: 3 predictions per minute
    
    Returns:
    - Core prediction data (like V1.0)
    - Trend analysis (short-term and long-term direction with strength)
    - Contextual tags (e.g., "low_volatility", "ranging", "bullish_crossover")
    - Trading suggestions with conviction levels and risk assessment
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
        
        # ===== V1.1 ENHANCEMENTS =====
        
        # Extract trend information
        latest_row = df_features.iloc[-1]
        
        # Determine short-term trend
        short_ma = latest_row.get('SMA_20', 0)
        mid_ma = latest_row.get('SMA_50', 0)
        if current_price > short_ma > mid_ma:
            short_term_direction = "bullish"
        elif current_price < short_ma < mid_ma:
            short_term_direction = "bearish"
        else:
            short_term_direction = "neutral"
        
        # Determine long-term trend
        long_ma = latest_row.get('SMA_200', 0)
        if current_price > long_ma and short_ma > long_ma:
            long_term_direction = "bullish"
        elif current_price < long_ma and short_ma < long_ma:
            long_term_direction = "bearish"
        else:
            long_term_direction = "neutral"
        
        # Calculate trend strength (0.0 to 1.0)
        macd = abs(latest_row.get('MACD', 0))
        rsi = latest_row.get('RSI_14', 50)
        rsi_strength = abs(rsi - 50) / 50  # 0 to 1
        adx = latest_row.get('ADX_14', 0) / 100 if latest_row.get('ADX_14', 0) <= 100 else 1.0
        trend_strength = float(np.mean([rsi_strength, adx]))
        
        trend_info = TrendInfo(
            short_term=short_term_direction,
            long_term=long_term_direction,
            strength=round(trend_strength, 2)
        )
        
        # Generate contextual tags
        tags = []
        
        # Volatility tags
        volatility = latest_row.get('volatility', 0)
        if volatility < 0.01:
            tags.append("low_volatility")
        elif volatility > 0.03:
            tags.append("high_volatility")
        
        # Momentum tags
        rsi_val = latest_row.get('RSI_14', 50)
        if rsi_val > 70:
            tags.append("overbought")
        elif rsi_val < 30:
            tags.append("oversold")
        
        # Trend tags
        bb_upper = latest_row.get('BB_upper', float('inf'))
        bb_lower = latest_row.get('BB_lower', 0)
        if current_price > bb_upper:
            tags.append("expansion")
        elif current_price < bb_lower:
            tags.append("contraction")
        else:
            tags.append("ranging")
        
        # Crossover tags
        macd_val = latest_row.get('MACD', 0)
        macd_signal = latest_row.get('MACD_signal', 0)
        if macd_val > macd_signal:
            tags.append("bullish_crossover")
        elif macd_val < macd_signal:
            tags.append("bearish_crossover")
        
        # Volume tags
        volume_sma = latest_row.get('volume_sma', 1)
        current_volume = latest_row.get('volume', 0)
        if current_volume > volume_sma * 1.5:
            tags.append("high_volume")
        elif current_volume < volume_sma * 0.5:
            tags.append("low_volume")
        
        # Generate trading suggestion
        # Score calculation
        technical_score = int(confidence * 100)
        
        # Momentum score
        momentum_score = int((rsi_strength + adx) * 50)
        
        # Volatility score (lower is better for confidence)
        volatility_score = int(max(0, 100 - (volatility * 1000)))
        
        # Confidence boost based on alignment
        alignment_boost = 0
        if short_term_direction == long_term_direction:
            alignment_boost += 10
        if trend_strength > 0.7:
            alignment_boost += 10
        
        # Total score
        total_score = technical_score + momentum_score + volatility_score + alignment_boost
        
        score_breakdown = ScoreBreakdown(
            confidence_boost=alignment_boost,
            trend_score=int(trend_strength * 100),
            total_score=total_score
        )
        
        # Determine action
        if prediction == 1 and confidence > 0.6:
            action = "BUY"
            risk = "medium" if confidence > 0.75 else "high"
            conviction_level = "high" if confidence > 0.75 else "medium"
            reasons = [
                f"Model predicts upward movement with {confidence*100:.1f}% confidence",
                f"Short-term trend is {short_term_direction}",
                f"RSI at {rsi_val:.1f}"
            ]
        elif prediction == 2 and confidence > 0.6:
            action = "SELL"
            risk = "medium" if confidence > 0.75 else "high"
            conviction_level = "high" if confidence > 0.75 else "medium"
            reasons = [
                f"Model predicts downward movement with {confidence*100:.1f}% confidence",
                f"Short-term trend is {short_term_direction}",
                f"RSI at {rsi_val:.1f}"
            ]
        elif confidence < 0.5:
            action = "WAIT"
            risk = "low"
            conviction_level = "low"
            reasons = [
                f"Low model confidence ({confidence*100:.1f}%)",
                "Market conditions unclear",
                "Consider waiting for better signal"
            ]
        else:
            action = "HOLD"
            risk = "low"
            conviction_level = "medium"
            reasons = [
                f"No strong directional signal (confidence: {confidence*100:.1f}%)",
                f"Trend alignment: {short_term_direction}/{long_term_direction}",
                "Monitor for clearer signals"
            ]
        
        suggestion = TradingSuggestion(
            action=action,
            conviction=conviction_level,
            reasoning=reasons,
            risk_level=risk,
            score_breakdown=score_breakdown
        )
        
        # Build V1.1 response
        response = PredictionResponseV1_1(
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
            trend=trend_info,
            tags=tags,
            suggestion=suggestion,
            model_version=metadata.get('version', 'unknown'),
            feature_count=len(feature_columns) if feature_columns else 0
        )
        
        return response
        
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
    
    # Determine model architecture
    use_stacking = metadata.get('use_stacking', False)
    has_meta_learner = metadata.get('has_meta_learner', False)
    
    if use_stacking and has_meta_learner:
        model_type = "Stacked Ensemble: (CatBoost + Random Forest + Logistic) → Meta-Learner"
        ensemble_description = "Base models generate predictions, meta-learner combines them for final output"
    else:
        model_type = "Weighted Ensemble (CatBoost + Random Forest + Logistic Regression)"
        ensemble_description = "Weighted average of base model predictions"
    
    return {
        "metadata": metadata,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "n_features": metadata.get('n_features', len(feature_columns) if feature_columns else 0),
        "sequence_length": sequence_length,
        "model_loaded": ensemble_model is not None,
        "symbol": metadata.get('symbol', 'N/A'),
        "interval": metadata.get('interval', 'N/A'),
        "training_date": metadata.get('training_date', 'N/A'),
        "model_type": model_type,
        "model_architecture": metadata.get('model_architecture', model_type),
        "use_stacking": use_stacking,
        "has_meta_learner": has_meta_learner,
        "ensemble_description": ensemble_description,
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
    
    # Create user in database (generates API key automatically)
    user = await db.create_user(email=email, name=name, quota_limit=1000)
    
    return {
        "success": True,
        "api_key": user["api_key"],
        "name": name,
        "email": email,
        "quota_limit": "1000 predictions per day",
        "usage": f"Include header: Authorization: Bearer {user['api_key']}"
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
        # Get user usage from database
        usage = await db.get_user_usage(auth_data["user_id"])
        
        return {
            "user_type": "authenticated",
            "name": auth_data["name"],
            "email": auth_data["email"],
            "quota_limit": f"{usage['quota_limit']} predictions per day",
            "requests_today": usage["requests_today"],
            "requests_remaining": usage["requests_remaining"]
        }


@app.delete("/api-keys/revoke")
async def revoke_api_key(
    email: str,
    admin_secret: str = Header(..., alias="X-Admin-Secret")
):
    """Revoke an API key by email (Admin only)"""
    ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change_this_secret_key")
    
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Find user by email
    user = await db.get_user_by_email(email)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Revoke the API key
    success = await db.revoke_api_key(user["_id"])
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to revoke API key")
    
    return {
        "success": True,
        "message": f"API key for {email} revoked successfully"
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
