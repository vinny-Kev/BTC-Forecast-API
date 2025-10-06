# 🤖 Bitcoin AI Prediction API

**Production-ready ML microservice for Bitcoin price movement predictions**

## 🚀 Live API

**Base URL**: `https://bitcoin-prediction-api.onrender.com` (after deployment)

## 📡 API Endpoints

### Health Check
```bash
GET /
GET /health
```

### Make Prediction
```bash
POST /predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "interval": "1m"
}
```

### Model Info
```bash
GET /model/info
```

## 🔑 Authentication (Future)

Currently open for demo. Premium API keys coming soon.

## 📊 Response Format

```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-10-06T12:00:00",
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

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Models**: CatBoost + LSTM + Random Forest Ensemble
- **Features**: 20 technical indicators
- **Data**: Live Binance API

## 💼 Contact

For API access, custom integrations, or enterprise solutions:
**kevinroymaglaqui29@gmail.com**

---

**Built with ❤️ by Kevin Roy Maglaqui**
