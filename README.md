# 🚀 Bitcoin AI Prediction API# 🤖 Bitcoin AI Prediction API



**High-performance Bitcoin price prediction API using optimized 3-model ensemble****Production-ready ML microservice for Bitcoin price movement predictions**



[![Accuracy](https://img.shields.io/badge/Accuracy-91.09%25-brightgreen)](https://btc-forecast-api.onrender.com)## 🚀 Live API

[![API Status](https://img.shields.io/badge/API-Live-brightgreen)](https://btc-forecast-api.onrender.com)

[![Models](https://img.shields.io/badge/Models-3--Ensemble-blue)]()**Base URL**: `https://bitcoin-prediction-api.onrender.com` (after deployment)



## 🎯 Overview## 📡 API Endpoints



This API provides real-time Bitcoin price movement predictions using a sophisticated ensemble of machine learning models:### Health Check

```bash

- **CatBoost (50%)** - Gradient boosting for structured dataGET /

- **Random Forest (25%)** - Feature importance and generalization  GET /health

- **Logistic Regression (25%)** - Linear baseline and fast inference```



**Performance**: 91.09% accuracy on test data with 0.3182 F1 score### Make Prediction

```bash

## 🚀 Quick StartPOST /predict

Content-Type: application/json

### API Endpoint

```{

https://btc-forecast-api.onrender.com  "symbol": "BTCUSDT",

```  "interval": "1m"

}

### Health Check```

```bash

curl https://btc-forecast-api.onrender.com/### Model Info

``````bash

GET /model/info

### Make Prediction```

```bash

curl -X POST https://btc-forecast-api.onrender.com/predict \## 🔑 Authentication (Future)

  -H "Content-Type: application/json" \

  -d '{"symbol": "BTCUSDT", "interval": "1m"}'Currently open for demo. Premium API keys coming soon.

```

## 📊 Response Format

## 📊 API Reference

```json

### GET `/`{

Health check endpoint  "symbol": "BTCUSDT",

```json  "timestamp": "2025-10-06T12:00:00",

{  "prediction": 1,

  "status": "healthy",  "prediction_label": "Large Upward Movement Expected",

  "model_loaded": true,  "confidence": 0.85,

  "timestamp": "2025-10-06T07:20:15.859554"  "probabilities": {

}    "no_movement": 0.10,

```    "large_up": 0.85,

    "large_down": 0.05

### POST `/predict`  },

Get Bitcoin price movement prediction  "current_price": 62500.00,

  "expected_movement": 0.5,

**Request:**  "next_periods": [...]

```json}

{```

  "symbol": "BTCUSDT",

  "interval": "1m"## 🛠️ Tech Stack

}

```- **Framework**: FastAPI

- **Models**: CatBoost + LSTM + Random Forest Ensemble

**Response:**- **Features**: 20 technical indicators

```json- **Data**: Live Binance API

{

  "symbol": "BTCUSDT",## 💼 Contact

  "prediction": 0,

  "prediction_label": "No Significant Movement",For API access, custom integrations, or enterprise solutions:

  "confidence": 0.8561,**kevinroymaglaqui29@gmail.com**

  "probabilities": {**kevinroymaglaqui.is-a.dev**

    "no_movement": 0.8561,

    "large_up": 0.0322,---

    "large_down": 0.1116

  },**Built with ❤️ by Kevin Roy Maglaqui** << Not trying to be cringe

  "current_price": 123944.92,**

  "next_periods": [...]
}
```

### GET `/model/info`
Get model information and performance metrics

## 🔧 Architecture

### Model Pipeline
```
Raw Price Data → Feature Engineering → 3-Model Ensemble → Prediction
```

### Features (70 total)
- **Price**: OHLCV, returns, moving averages
- **Technical**: RSI, MACD, Bollinger Bands, ATR
- **Volume**: Volume indicators, VWAP
- **Time**: Hour, day patterns
- **Market**: Volatility, momentum indicators

### Prediction Classes
- **0**: No Significant Movement (< 0.5% change)
- **1**: Large Upward Movement (> 0.5% up)
- **2**: Large Downward Movement (> 0.5% down)

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- Binance API keys (for data fetching)

### Setup
```bash
git clone https://github.com/vinny-Kev/BTC-Forecast-API.git
cd BTC-Forecast-API
pip install -r requirements.txt
```

### Environment Variables
```bash
cp .env.example .env
# Add your Binance API keys to .env
```

### Run Locally
```bash
uvicorn prediction_api:app --reload
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 91.09% |
| **F1 Score** | 0.3182 |
| **Response Time** | < 2 seconds |
| **Uptime** | 99.9% |

### Model Improvements
- ✅ Removed LSTM complexity (+1.88% accuracy)
- ✅ Optimized ensemble weights
- ✅ Enhanced feature engineering
- ✅ Reduced overfitting

## 🚀 Deployment

### Render (Production)
- Automatically deploys from `main` branch
- Environment: Python 3.11
- Health checks enabled
- Auto-scaling enabled

### Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
catboost==1.2.3
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.25.2
python-binance==1.0.19
python-dotenv==1.0.1
joblib==1.3.2
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📞 Support

- **API Issues**: Check health endpoint first
- **Performance**: Model retrains automatically with new data
- **Rate Limits**: Fair use policy applied

---

**Live API**: [https://btc-forecast-api.onrender.com](https://btc-forecast-api.onrender.com)