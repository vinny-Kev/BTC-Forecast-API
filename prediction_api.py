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
from typing import List, Dict, Optional
from datetime import datetime
import secrets
import json
import time
import pickle
import logging
import tensorflow as tf
import uuid
import bcrypt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from feature_engineering import FeatureEngineer
from models import EnsembleModel
from database import db  # MongoDB database instance
from src.model_manager import ModelManager

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
transformer_model = None
transformer_metadata = None
# Model manager for safe load/validate/swap
model_manager = ModelManager()

# Class labels for ensemble predictions
ENSEMBLE_LABELS = {
    0: "No Significant Movement",
    1: "Large Upward Movement Expected",
    2: "Large Downward Movement Expected"
}

# Security
security = HTTPBearer(auto_error=False)

# Logger
logger = logging.getLogger("prediction_api")
logger.setLevel(logging.INFO)

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

        # Allow disabling the guest limiter locally for testing by setting
        # the environment variable DISABLE_GUEST_LIMITS=true
        disable_limiter = os.getenv("DISABLE_GUEST_LIMITS", "false").lower() in ("1", "true", "yes")

        if not disable_limiter:
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
        else:
            # Limiter disabled: allow guest requests without incrementing usage
            return {
                "type": "guest",
                "ip": client_ip,
                "calls_used": guest_usage.get("calls_used", 0),
                "calls_remaining": float("inf")
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


class TransformerPredictionResponse(PredictionResponse):
    """Response model for transformer regression predictions."""
    prediction: float
    probabilities: Dict[str, float] = Field(default_factory=dict)


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


def validate_metadata(metadata):
    """Validate metadata to ensure all required fields are present."""
    # Only check for fields that should be in metadata.json
    # feature_columns, preprocessor, model_type are loaded separately
    required_fields = ["sequence_length"]
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        logger.warning(
            "⚠ Metadata is missing expected fields: %s. Falling back to defaults where necessary.",
            ", ".join(missing_fields)
        )
        return False
    return True


def load_models(model_dir: str):
    """Load trained models, preprocessor, and transformer assets."""
    global ensemble_model
    global preprocessor
    global feature_engineer
    global feature_columns
    global metadata
    global sequence_length
    global transformer_model
    global transformer_metadata

    logger.info(f"Loading models from {model_dir}...")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    logger.info(f"Directory contents: {os.listdir(model_dir)}")

    # ------------------------------------------------------------------
    # Ensemble models (CatBoost + Random Forest + optional Logistic)
    # ------------------------------------------------------------------
    ensemble_required = [
        os.path.join(model_dir, 'catboost_model.cbm'),
        os.path.join(model_dir, 'random_forest_model.pkl')
    ]

    if all(os.path.exists(path) for path in ensemble_required):
        try:
            ensemble_model = EnsembleModel(n_classes=3)
            ensemble_model.load(model_dir)
            logger.info("✓ Ensemble model loaded")
        except Exception as e:
            logger.warning(f"⚠ Failed to load ensemble model: {e}")
            ensemble_model = None
    else:
        missing = [os.path.basename(path) for path in ensemble_required if not os.path.exists(path)]
        logger.info(
            "ℹ️ Skipping ensemble model load (missing files: %s)",
            ", ".join(missing) if missing else "unknown"
        )
        ensemble_model = None

    # ------------------------------------------------------------------
    # Metadata (best-effort, tolerate missing fields)
    # ------------------------------------------------------------------
    metadata = {}
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if validate_metadata(metadata):
                logger.info("✓ Metadata validated for ensemble model")
        except Exception as e:
            logger.warning(f"⚠ Failed to load metadata: {e}")
            metadata = {}
    else:
        logger.info(f"ℹ️ Metadata file not found at {metadata_path}; using defaults")

    # Defaults to ensure downstream code has required keys
    metadata.setdefault('sequence_length', 30)
    metadata.setdefault('threshold_percent', 0.5)
    metadata.setdefault('feature_columns', [])
    metadata.setdefault('model_type', 'unknown')
    preprocessor_candidates = []
    if metadata.get('preprocessor'):
        preprocessor_candidates.append(metadata['preprocessor'])
    preprocessor_candidates.extend(['preprocessor.pkl', 'scaler.pkl'])

    # ------------------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------------------
    feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
    if os.path.exists(feature_columns_path):
        try:
            with open(feature_columns_path, 'rb') as f:
                feature_columns = pickle.load(f)
            logger.info(f"✓ Feature columns loaded ({len(feature_columns)} features)")
        except Exception as e:
            logger.warning(f"⚠ Could not load feature columns: {e}")
            feature_columns = metadata.get('feature_columns', [])
    else:
        feature_columns = metadata.get('feature_columns', [])
        if feature_columns:
            logger.info("ℹ️ Using feature columns embedded in metadata")
        else:
            logger.warning("⚠ Feature columns file not found; predictions may fail if columns mismatch")

    metadata['feature_columns'] = feature_columns

    # ------------------------------------------------------------------
    # Preprocessor / scaler (try multiple filenames)
    # ------------------------------------------------------------------
    preprocessor = None
    for candidate in preprocessor_candidates:
        candidate_path = candidate if os.path.isabs(candidate) else os.path.join(model_dir, candidate)
        if os.path.exists(candidate_path):
            try:
                preprocessor = joblib.load(candidate_path)
                logger.info(f"✓ Preprocessor loaded from {candidate_path}")
                break
            except Exception:
                try:
                    with open(candidate_path, 'rb') as f:
                        preprocessor = pickle.load(f)
                    logger.info(f"✓ Preprocessor loaded from {candidate_path} (pickle fallback)")
                    break
                except Exception as e:
                    logger.warning(f"⚠ Failed to load preprocessor from {candidate_path}: {e}")
    if preprocessor is None:
        logger.warning("⚠ No preprocessor/scaler found; predictions will not be available until provided")

    # ------------------------------------------------------------------
    # Transformer model (new v2 endpoint)
    # ------------------------------------------------------------------
    try:
        load_transformer_model(model_dir)
    except FileNotFoundError as e:
        logger.info(f"ℹ️ Transformer assets not found: {e}")
    except Exception as e:
        logger.warning(f"⚠ Failed to load transformer model: {e}")

    if transformer_metadata:
        sequence_length = transformer_metadata.get('sequence_length', metadata.get('sequence_length', 30))
    else:
        sequence_length = metadata.get('sequence_length', 30)

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    logger.info("✓ Feature engineer initialized")

    logger.info("✓ Model loading completed!")


def load_transformer_model(model_dir: str):
    """Load the transformer regressor v2 model."""
    global transformer_model, transformer_metadata

    logger.info(f"Loading transformer model via ModelManager from {model_dir}...")

    # Use ModelManager to load/validate/swap the transformer candidate
    try:
        model_manager.reload_latest(model_dir)
        active = model_manager.get_active()
        if active:
            transformer_model = active.model
            transformer_metadata = active.metadata
            # Ensure feature_columns is populated if manager provided them
            if active.feature_columns:
                # assign to the global feature_columns variable for backward compatibility
                try:
                    global feature_columns
                    feature_columns = active.feature_columns
                except Exception:
                    pass
            logger.info(f"\u2713 Transformer model loaded and active from {active.path}")
        else:
            raise RuntimeError("ModelManager did not produce an active model")
    except Exception as e:
        logger.error(f"Failed to load transformer model via ModelManager: {e}")
        raise


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
    
    logger.info(f"Starting up API...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Connect to MongoDB - REQUIRED for security
    try:
        await db.connect()
        logger.info("✓ MongoDB connection established")
    except Exception as e:
        logger.warning(f"⚠ MongoDB connection failed: {e}")
        logger.warning("⚠ WARNING: API will reject all requests until database is connected")
        logger.warning("⚠ This is intentional for security - authentication requires database")
    
    # Try to load the latest model
    model_dir = get_latest_model_dir()
    
    if model_dir:
        logger.info(f"Found model directory: {model_dir}")
        try:
            load_models(model_dir)
            logger.info(f"✓ API ready with models from: {model_dir}")
        except Exception as e:
            logger.warning(f"⚠ Warning: Could not load models: {e}")
            logger.warning("API will run without loaded models. Train models first.")
            ensemble_model = None
    else:
        logger.warning("⚠ No models found in any of the expected locations:")
        logger.warning("  - ./data/models")
        logger.warning("  - data/models") 
        logger.warning(f"  - {os.path.join(os.path.dirname(__file__), 'data', 'models')}")
        logger.warning("Please train models first using the training pipeline.")
        ensemble_model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from MongoDB on shutdown"""
    await db.disconnect()
    logger.info("✓ Disconnected from MongoDB")


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


# Replace hardcoded ADMIN_SECRET with environment variable
ADMIN_SECRET = os.getenv("ADMIN_SECRET")
if not ADMIN_SECRET:
    raise RuntimeError("ADMIN_SECRET environment variable is not set")


# Refactor shared logic in /predict and /v1.1/predict endpoints
# Extract feature generation and prediction logic into a helper function
def generate_features_and_predict(df, context):
    """Helper function to generate features and make predictions."""
    df_features = feature_engineer.generate_all_features(
        df,
        order_book_context=context.get('order_book'),
        ticker_context=context.get('ticker'),
        create_targets=False
    )

    # Ensure feature_columns exist and handle missing columns gracefully
    if not feature_columns:
        raise RuntimeError("Feature columns are not available for prediction")

    # Reindex to the expected feature columns, filling missing columns with zeros
    X_all = df_features.reindex(columns=feature_columns).fillna(0)

    # Ensure we have at least sequence_length + 1 rows; if not, pad with zeros at the top
    required_rows = sequence_length + 1
    if len(X_all) < required_rows:
        pad_rows = required_rows - len(X_all)
        pad_df = pd.DataFrame(0, index=range(pad_rows), columns=feature_columns)
        X_all = pd.concat([pad_df, X_all], ignore_index=True)

    X = X_all.tail(required_rows)
    X_scaled = preprocessor.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)

    X_regular = X_scaled_df.tail(1).values
    probabilities = ensemble_model.predict_proba(X_regular)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    return prediction, confidence, probabilities, df_features


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    auth_data: dict = Depends(verify_api_key)
):
    if ensemble_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )

    try:
        from data_scraper import DataScraper
        scraper = DataScraper(symbol=prediction_request.symbol, interval=prediction_request.interval)
        df = scraper.fetch_historical_("6 hours ago UTC")
        _, context = scraper.fetch_context_data()

        prediction, confidence, probabilities, df_features = generate_features_and_predict(df, context)

        prediction_label = ENSEMBLE_LABELS.get(prediction, "Unknown Movement")
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
        logger.error(f"Prediction failed: {e}")
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
        
        # Get only the feature columns we trained on, handling missing columns
        if not feature_columns:
            raise HTTPException(status_code=503, detail="Feature columns not available")

        X_all = df_features.reindex(columns=feature_columns).fillna(0)

        # Ensure enough rows; pad at the top if necessary
        required_rows = sequence_length + 1
        if len(X_all) < required_rows:
            pad_rows = required_rows - len(X_all)
            pad_df = pd.DataFrame(0, index=range(pad_rows), columns=feature_columns)
            X_all = pd.concat([pad_df, X_all], ignore_index=True)

        X = X_all.tail(required_rows)

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
        
        # Define current_price and prediction_label
        current_price = float(df['close'].iloc[-1])
        prediction_label = ENSEMBLE_LABELS.get(prediction, "Unknown Movement")
        
        # Determine short-term trend
        latest_row = df_features.iloc[-1]
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

@app.post("/v2/predict", response_model=PredictionResponse)
async def predict_v2(
    request: Request,
    prediction_request: PredictionRequest,
    auth_data: dict = Depends(verify_api_key)
):
    """
    Get predictions using the updated transformer model.
    """
    if transformer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Transformer model not loaded. Please train and load the model first."
        )

    try:
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

        # Ensure feature columns exist and handle missing columns
        if not feature_columns:
            raise HTTPException(status_code=503, detail="Feature columns not available")

        seq_len = int(transformer_metadata.get('sequence_length', sequence_length))
        X_all = df_features.reindex(columns=feature_columns).fillna(0)

        # Pad if necessary
        if len(X_all) < seq_len:
            pad_rows = seq_len - len(X_all)
            pad_df = pd.DataFrame(0, index=range(pad_rows), columns=feature_columns)
            X_all = pd.concat([pad_df, X_all], ignore_index=True)

        X = X_all.tail(seq_len)
        X_scaled = preprocessor.transform(X)

        # Make predictions
        predictions = transformer_model.predict(X_scaled)
        prediction = float(predictions[-1])  # Use the last prediction

        # Define current_price and prediction_label
        current_price = float(df['close'].iloc[-1])
        prediction_label = "Transformer Prediction"

        return PredictionResponse(
            symbol=prediction_request.symbol,
            timestamp=datetime.now().isoformat(),
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=1.0,  # Placeholder for confidence
            probabilities={},  # Placeholder for probabilities
            current_price=current_price,
            expected_movement=None
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/v2/transformermodel/keras", response_model=TransformerPredictionResponse)
async def predict_transformer_keras(
    request: Request,
    prediction_request: PredictionRequest,
    auth_data: dict = Depends(verify_api_key)
):
    """
    V2 endpoint specifically for the custom transformer Keras model.
    Keeps the older endpoints intact; this one targets the model exported
    as `transformer_regressor_v2_model.keras` and uses the transformer
    metadata for sequence length.
    """
    if transformer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Transformer Keras model not loaded. Please train and load the model first."
        )

    try:
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

        seq_len = int(transformer_metadata.get('sequence_length', sequence_length))

        if not feature_columns:
            raise HTTPException(status_code=503, detail="Feature columns not available for transformer prediction")

        # Reindex to expected features and fill missing columns with zeros
        X_all = df_features.reindex(columns=feature_columns).fillna(0)

        # Pad history if shorter than sequence length
        if len(X_all) < seq_len:
            pad_rows = seq_len - len(X_all)
            pad_df = pd.DataFrame(0, index=range(pad_rows), columns=feature_columns)
            X_all = pd.concat([pad_df, X_all], ignore_index=True)

        X = X_all.tail(seq_len)
        X_scaled = preprocessor.transform(X)

        # Model expects batch dimension
        import numpy as _np
        if X_scaled.ndim == 2:
            X_input = _np.expand_dims(X_scaled, axis=0)
        else:
            X_input = X_scaled

        preds = transformer_model.predict(X_input)

        # Normalize handling of different output shapes
        if hasattr(preds, 'shape'):
            if preds.ndim == 3:
                # e.g., (batch, seq, 1)
                pred_value = float(preds[0, -1, 0])
            elif preds.ndim == 2:
                # e.g., (batch, 1)
                pred_value = float(preds[0, -1]) if preds.shape[1] > 1 else float(preds[0, 0])
            elif preds.ndim == 1:
                pred_value = float(preds[0])
            else:
                pred_value = float(preds.flatten()[-1])
        else:
            pred_value = float(preds)

        current_price = float(df['close'].iloc[-1])

        return TransformerPredictionResponse(
            symbol=prediction_request.symbol,
            timestamp=datetime.now().isoformat(),
            prediction=pred_value,
            prediction_label="TransformerModel-Keras-v2",
            confidence=1.0,
            probabilities={},
            current_price=current_price,
            expected_movement=None
        )

    except Exception as e:
        logger.error(f"Transformer Keras prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transformer prediction failed: {e}")


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
        # Model considered loaded if either ensemble or transformer is available
        "model_loaded": (ensemble_model is not None) or (transformer_model is not None),
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
    
    # If user already exists, rotate their API key instead of inserting (avoid DuplicateKeyError)
    existing = await db.db.users.find_one({"email": email})
    if existing:
        # generate new api key and update the stored hash
        new_api_key = f"btc_{uuid.uuid4().hex}"
        new_hash = bcrypt.hashpw(new_api_key.encode(), bcrypt.gensalt()).decode()
        await db.db.users.update_one(
            {"_id": existing["_id"]},
            {"$set": {"api_key_hash": new_hash, "name": name, "quota_limit": 1000, "is_active": True}}
        )
        return {
            "success": True,
            "api_key": new_api_key,
            "name": name,
            "email": email,
            "quota_limit": "1000 predictions per day",
            "usage": f"Include header: Authorization: Bearer {new_api_key}"
        }

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
    
    logger.info("\n" + "="*60)
    logger.info("Starting Crypto Price Movement Prediction API")
    logger.info("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
