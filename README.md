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
- **CatBoost** - Gradient boosting
- **Scikit-learn** - Random Forest, Logistic Regression, Preprocessing
- **TensorFlow/Keras** - Meta-learner LSTM

### Data & Features
- **Binance API** - Live market data
- **20 Technical Indicators** - Engineered features
- **Real-time Preprocessing** - RobustScaler for outlier handling

---

## 🔧 Architecture

### Data Pipeline
```
Binance API → Feature Engineering → RobustScaler → Ensemble Models → Meta-Learner → Prediction
```

### Technical Features (20 total)
- **Volatility**: ATR, ATR%, volatility_5/10/20/50
- **Volume**: Volume MA (7/14/21), OBV
- **Price Patterns**: BB width, HL range, price-to-SMA ratios
- **Momentum**: MACD, MACD signal, ROC
- **Returns**: 50-period returns
- **Time**: Hour cosine encoding

### Prediction Classes
- **0**: No Significant Movement (< 0.2% change in 6 periods)
- **1**: Large Upward Movement (> 0.2% up in 6 periods)
- **2**: Large Downward Movement (> 0.2% down in 6 periods)

---

## 🚀 Local Development

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (or local MongoDB)
- Binance API keys

### Setup
```bash
# Clone repository
git clone https://github.com/vinny-Kev/BTC-Forecast-API.git
cd BTC-Forecast-API

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your credentials
```

### Environment Variables (.env)
```bash
# Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# MongoDB
MONGODB_URL=mongodb+srv://<user>:<password>@cluster0.xxx.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=btc_prediction_api

# Admin Secret (for API key management)
ADMIN_SECRET=your_admin_secret_key
```

### Run API Locally
```bash
python prediction_api.py
# API runs on http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## 📈 Performance Metrics

### Model Performance
| Metric | Train | Test |
|--------|-------|------|
| **Accuracy** | 82.90% | 65.76% |
| **Precision (Macro)** | 49.40% | 40.60% |
| **Recall (Macro)** | 87.14% | 47.34% |
| **F1 Score (Macro)** | 55.52% | 39.48% |
| **ROC AUC (OvR)** | 0.9648 | 0.7097 |

### Overfitting Analysis
- Train-Test Accuracy Gap: **17.15%**
- Train-Test F1 Gap: **16.04%**
- Train-Test ROC Gap: **25.51%**

*Note: Some overfitting present but ROC AUC of 0.71 indicates reasonable generalization*

### API Performance
- **Response Time**: < 2 seconds (live data fetch + inference)
- **Uptime Target**: 99.9%
- **Auto-scaling**: Enabled on Render

---

## 🚢 Deployment

### Render Configuration
```yaml
services:
  - type: web
    name: btc-prediction-api
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn prediction_api:app --host 0.0.0.0 --port $PORT
```

### Required Environment Variables (Render)
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `MONGODB_URL` (MongoDB Atlas connection string)
- `DATABASE_NAME`
- `ADMIN_SECRET`

### Git LFS for Model Files
Model files are tracked with Git LFS:
```bash
git lfs track "*.cbm"
git lfs track "*.pkl"
git lfs track "*.h5"
```

---

## 🗄️ Database Schema

### Users Collection
```javascript
{
  _id: "uuid",
  email: "user@example.com",
  name: "User Name",
  api_key_hash: "bcrypt_hash",
  created_at: ISODate("2025-10-08T12:00:00Z"),
  requests_today: 42,
  quota_limit: 1000,
  last_quota_reset: ISODate("2025-10-08T00:00:00Z"),
  is_active: true
}
```

### Guest Usage Collection
```javascript
{
  ip_address: "192.168.1.1",
  calls_used: 2,
  calls_limit: 3,
  first_call: ISODate("2025-10-08T12:00:00Z"),
  last_reset: ISODate("2025-10-08T12:00:00Z")
}
```

### Predictions Collection
```javascript
{
  _id: "uuid",
  user_id: "user_uuid" | "guest",
  ip_address: "192.168.1.1",
  timestamp: ISODate("2025-10-08T12:00:00Z"),
  symbol: "BTCUSDT",
  interval: "1m",
  prediction: 1,
  confidence: 0.85,
  model_version: "1.0.0",
  latency_ms: 1234.5,
  result: { ... }
}
```

---

## 📝 API Documentation

Interactive Swagger documentation available at:
- **Local**: http://localhost:8000/docs
- **Production**: https://btc-forecast-api.onrender.com/docs

---

## 💼 Contact & Support

**Developer**: Kevin Roy Maglaqui

- **Email**: kevinroymaglaqui27@gmail.com
- **Portfolio**: [kevinroymaglaqui.is-a.dev](https://kevinroymaglaqui.is-a.dev)
- **GitHub**: [@vinny-Kev](https://github.com/vinny-Kev)

### Enterprise Solutions
For custom integrations, higher rate limits, or white-label solutions, please reach out!

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🎯 Roadmap

- [x] Stacked ensemble with meta-learner
- [x] MongoDB integration for persistence
- [x] V1.1 API with enhanced predictions
- [x] API key management system
- [ ] WebSocket support for real-time predictions
- [ ] Multi-symbol support (ETH, SOL, etc.)
- [ ] Historical prediction accuracy tracking
- [ ] Custom alert system

---

**Built with ❤️ by Kevin Roy Maglaqui**

*Last Updated: October 8, 2025*
