# 🤖 Bitcoin AI Prediction API

**Production-ready ML microservice for Bitcoin price movement predictions with stacked ensemble architecture**

[![API Status](https://img.shields.io/badge/API-Live-brightgreen)](https://btc-forecast-api.onrender.com)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-65.76%25-blue)]()
[![ROC AUC](https://img.shields.io/badge/ROC_AUC-0.7097-orange)]()

**Base URL**: `https://btc-forecast-api.onrender.com`

---

## 🎯 Overview

Real-time Bitcoin price movement predictions using a sophisticated **stacked ensemble** of machine learning models with MongoDB-backed persistence.

### Model Architecture

**Stacking Ensemble**: Base Models → Meta-Learner → Final Prediction

- **CatBoost (50%)** - Gradient boosting for structured data
- **Random Forest (25%)** - Feature importance and generalization  
- **Logistic Regression (25%)** - Linear baseline for fast inference
- **Meta-Learner (LSTM)** - Stacking layer that combines base predictions

**Training Data**: 50,000 samples | **Test Accuracy**: 65.76% | **ROC AUC**: 0.7097

---

## 📡 API Endpoints

### 🏥 Health Check
```bash
GET /
GET /health
```

### 🔮 Predictions

#### V1.0 - Basic Prediction
```bash
POST /predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "interval": "1m"
}
```

#### V1.1 - Enhanced Prediction (Recommended)
```bash
POST /v1.1/predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "interval": "1m"
}
```

**V1.1 Features**:
- ✨ Trend analysis (short-term & long-term)
- 🏷️ Contextual market tags (volatility, momentum, volume)
- 💡 Trading suggestions with conviction levels
- 🎯 Risk assessment
- 📊 Score breakdown

### 📊 Model Information
```bash
GET /model/info
```

### 🔑 API Key Management

#### Generate API Key (Admin Only)
```bash
POST /api-keys/generate
X-Admin-Secret: <admin_secret>
Content-Type: application/json

{
  "name": "User Name",
  "email": "user@example.com"
}
```

#### Check Usage
```bash
GET /api-keys/usage
Authorization: Bearer <api_key>
```

#### Revoke API Key (Admin Only)
```bash
DELETE /api-keys/revoke?email=user@example.com
X-Admin-Secret: <admin_secret>
```

---

## 📊 Response Examples

### V1.0 Response
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-10-08T12:00:00",
  "prediction": 1,
  "prediction_label": "Large Upward Movement Expected",
  "confidence": 0.85,
  "probabilities": {
    "no_movement": 0.10,
    "large_up": 0.85,
    "large_down": 0.05
  },
  "current_price": 62500.00,
  "expected_movement": 0.5,
  "next_periods": [...]
}
```

### V1.1 Response (Enhanced)
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-10-08T12:00:00",
  "prediction": 1,
  "prediction_label": "Large Upward Movement Expected",
  "confidence": 0.85,
  "probabilities": {...},
  "current_price": 62500.00,
  "trend": {
    "short_term": "bullish",
    "long_term": "bullish",
    "strength": 0.75
  },
  "tags": [
    "high_volatility",
    "bullish_crossover",
    "high_volume"
  ],
  "suggestion": {
    "action": "BUY",
    "conviction": "high",
    "reasoning": [
      "Model predicts upward movement with 85.0% confidence",
      "Short-term trend is bullish",
      "RSI at 65.2"
    ],
    "risk_level": "medium",
    "score_breakdown": {
      "confidence_boost": 20,
      "trend_score": 75,
      "total_score": 285
    }
  },
  "api_version": "1.1",
  "model_version": "1.0.0",
  "feature_count": 20
}
```

---

## 🔐 Authentication & Rate Limiting

### Guest Users (No API Key)
- ✅ **3 free predictions** per IP address
- ✅ Persistent tracking (survives refreshes)
- ✅ Stored in MongoDB

### Authenticated Users (With API Key)
- ✅ **1,000 predictions per day**
- ✅ Daily quota reset
- ✅ Usage tracking via `/api-keys/usage`
- ✅ MongoDB-backed persistence

### How to Get an API Key
Contact Kevin Maglaqui: **kevinroymaglaqui27@gmail.com**

---

## 🛠️ Tech Stack

### Core Framework
- **FastAPI** - High-performance async web framework
- **MongoDB Atlas** - Cloud database for persistence
- **Motor** - Async MongoDB driver

### Machine Learning
# BTC Forecast API — Minimal Deployment README

This repo contains a FastAPI microservice that serves ML predictions for Bitcoin price movement.

Purpose of this cleanup commit:
- Remove temporary/debug files
- Keep a single canonical README with deployment and local run instructions
- Ensure requirements.txt contains runtime deps
- Tighten .gitignore to avoid committing secrets and large models

Quick start (local):

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy and edit environment variables:

```powershell
copy .env.example .env
# edit .env and add your MONGODB_URL, BINANCE keys (if used), and ADMIN_SECRET
```

3. Run the API locally:

```powershell
python -m uvicorn prediction_api:app --host 127.0.0.1 --port 8000
```

Endpoints of interest:
- GET / or /health — health
- GET /model/info — model metadata and loaded state
- POST /v2/transformermodel/keras — transformer model regression (returns float prediction)
- POST /v1.1/predict — enriched ensemble predictions
- POST /api-keys/generate — admin-only; requires X-Admin-Secret header

Security & deployment notes:
- Do NOT commit .env or any secret values. Use `.env.example` as template.
- Model artifacts (large .cbm/.pkl/.keras files) should be deployed via Git LFS or a model artifact store; they are excluded by .gitignore.
- Ensure `MONGODB_URL` points to your Atlas or managed DB for production.
- Set `ADMIN_SECRET` in the environment for admin operations.

If you'd like, I can:
- Add a CI workflow to build & test the container
- Add an admin `/model/deploy` endpoint that accepts a model archive and atomically swaps it via `ModelManager`

---

Last updated: 2025-10-10
