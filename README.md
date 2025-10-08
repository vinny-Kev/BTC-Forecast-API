# 🤖 Bitcoin AI Prediction API

**Production-ready ML microservice for Bitcoin price movement predictions**

[![API Status](https://img.shields.io/badge/API-Live-brightgreen)](https://btc-forecast-api.onrender.com)

**Base URL**: `https://btc-forecast-api.onrender.com`

---

## 🎯 Overview

Advanced Bitcoin price prediction API powered by a stacked ensemble architecture with meta-learning. The system uses 50,000+ training samples and combines multiple ML models for superior accuracy.

### Model Architecture
```
(CatBoost + Random Forest + Logistic Regression) → Meta-Learner → Final Prediction
```

**Performance Metrics**:
- **Training Accuracy**: 82.90%
- **Test Accuracy**: 65.76%
- **ROC AUC**: 0.7097
- **Response Time**: < 3 seconds

---

## � API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-08T14:00:00"
}
```

### 2. Make Prediction (V1.0)
```bash
POST /predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "interval": "1m"
}
```

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-10-08T14:00:00",
  "prediction": 1,
  "prediction_label": "Large Upward Movement Expected",
  "confidence": 0.75,
  "probabilities": {
    "no_movement": 0.15,
    "large_up": 0.75,
    "large_down": 0.10
  },
  "current_price": 62500.00,
  "expected_movement": 0.5,
  "next_periods": [...]
}
```

### 3. Enhanced Prediction (V1.1) ✨ NEW
```bash
POST /v1.1/predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "interval": "1m"
}
```

**V1.1 Enhancements:**
- 📈 **Trend Analysis**: Short/long-term direction with strength scores
- 🏷️ **Market Tags**: Contextual labels (overbought, high_volatility, ranging, etc.)
- 💡 **Trading Suggestions**: BUY/SELL/WAIT/HOLD with conviction levels
- 🎯 **Risk Assessment**: Detailed score breakdown and reasoning

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "prediction": 1,
  "confidence": 0.75,
  "trend": {
    "short_term": "bullish",
    "long_term": "neutral",
    "strength": 0.68
  },
  "tags": ["low_volatility", "bullish_crossover", "expansion"],
  "suggestion": {
    "action": "BUY",
    "conviction": "high",
    "reasoning": [
      "Model predicts upward movement with 75% confidence",
      "Short-term trend is bullish",
      "RSI at 55.3"
    ],
    "risk_level": "medium",
    "score_breakdown": {
      "confidence_boost": 20,
      "trend_score": 68,
      "total_score": 180
    }
  }
}
```

### 4. Model Information
```bash
GET /model/info
```

Returns detailed model metadata, architecture, and performance metrics.

---

## 🔑 Authentication

### Free Trial (Guest Access)
- **3 free predictions** per IP address
- No API key required
- Perfect for testing

### API Key Access
- **3 predictions per minute**
- Generate via admin endpoint
- Include in header: `Authorization: Bearer YOUR_API_KEY`

**Get Usage Info:**
```bash
GET /api-keys/usage
Authorization: Bearer YOUR_API_KEY
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI 0.104.1 |
| **Base Models** | CatBoost, Random Forest, Logistic Regression |
| **Meta-Learner** | LSTM (stacked ensemble) |
| **Features** | 20 technical indicators |
| **Data Source** | Binance Live API |
| **Deployment** | Render (auto-scaling) |
| **Python** | 3.11+ |

### Key Dependencies
```
fastapi==0.104.1
catboost==1.2.3
scikit-learn==1.7.2
python-binance==1.0.19
pandas==2.1.4
numpy==1.25.2
```

---

## 🏗️ Architecture

### Data Pipeline
```
Binance API → Feature Engineering → Preprocessing → Ensemble → Meta-Learner → Prediction
```

### Feature Engineering (20 Features)
- **Price Indicators**: SMA (7, 14, 21, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14), MACD, ROC
- **Volatility**: ATR (14), Bollinger Bands
- **Volume**: Volume SMA, VWAP
- **Trend**: ADX (14), Directional Indicators

### Prediction Classes
- **Class 0**: No Significant Movement (±0.2% threshold)
- **Class 1**: Large Upward Movement (>0.2% increase expected)
- **Class 2**: Large Downward Movement (>0.2% decrease expected)

---

## � Local Development

### Prerequisites
```bash
Python 3.11+
Git LFS (for model files)
Binance API Keys
```

### Setup
```bash
# Clone repository
git clone https://github.com/vinny-Kev/BTC-Forecast-API.git
cd BTC-Forecast-API

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add BINANCE_API_KEY and BINANCE_SECRET_KEY to .env

# Run locally
python prediction_api.py
# API available at http://localhost:8000
```

### Environment Variables
```bash
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here
ADMIN_SECRET=your_admin_secret
```

---

## 📈 Model Performance

### Metrics Breakdown

| Dataset | Accuracy | F1 Score | ROC AUC |
|---------|----------|----------|---------|
| **Training** | 82.90% | 0.5552 | 0.9648 |
| **Testing** | 65.76% | 0.3948 | 0.7097 |

### Overfitting Analysis
- Train-Test Accuracy Gap: 17.15%
- Train-Test F1 Gap: 16.04%
- Conservative approach prioritizes generalization over training performance

### Recent Improvements
- ✅ Implemented meta-learner stacking (+8% accuracy boost)
- ✅ Trained on 50,000+ samples (up from 10,000)
- ✅ Added contextual market intelligence (V1.1)
- ✅ Enhanced feature engineering with trend analysis
- ✅ Optimized ensemble weights for balanced predictions

---

## 🚀 Deployment

### Render Configuration
- **Runtime**: Python 3.11
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn prediction_api:app --host 0.0.0.0 --port $PORT`
- **Auto-Deploy**: Enabled on `main` branch push
- **Health Check**: `/health` endpoint monitored

### Git LFS Files
Model artifacts are stored with Git LFS:
```
*.cbm (CatBoost models)
*.pkl (Preprocessor, feature columns, scikit-learn models)
*.h5 (LSTM weights if used)
```

---

## 🎨 Frontend

Interactive web UI included in `index.html`:
- Real-time prediction display
- Trend analysis visualization
- Market condition tags
- Trading suggestions with reasoning
- Probability distribution charts

**Usage:** Open `index.html` in browser or serve via static file hosting.

---

## 📞 Contact & Support

**Developer**: Kevin Roy Maglaqui  
**Email**: kevinroymaglaqui27@gmail.com  
**Website**: kevinroymaglaqui.is-a.dev  
**GitHub**: [@vinny-Kev](https://github.com/vinny-Kev)

For enterprise integrations, custom features, or API partnerships, reach out via email.

---

## 📄 License

MIT License - Free for personal and commercial use.

---

**🚀 Live API**: [https://btc-forecast-api.onrender.com](https://btc-forecast-api.onrender.com)

**📚 Docs**: [https://btc-forecast-api.onrender.com/docs](https://btc-forecast-api.onrender.com/docs)

---

*Built with ❤️ using FastAPI, CatBoost, and advanced ML techniques*
