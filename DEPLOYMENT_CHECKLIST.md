# Render Deployment Checklist ✅

## Environment Variables Required on Render

Make sure these are set in your Render dashboard (Environment tab):

### ✅ Required
- [x] `MONGODB_URL` - MongoDB Atlas connection string
  - Format: `mongodb+srv://<user>:<password>@cluster0.zdrqsde.mongodb.net/?retryWrites=true&w=majority`
  - ⚠️ Password must be URL-encoded if it contains special characters
  
- [x] `BINANCE_API_KEY` - Your Binance API key for fetching live data
- [x] `BINANCE_API_SECRET` - Your Binance API secret

### ⚙️ Optional
- [ ] `DATABASE_NAME` - Database name (default: `btc_prediction_api`)
- [ ] `ADMIN_SECRET` - Secret key for admin operations

---

## Post-Deployment Tests

### 1. Health Check
```bash
curl https://your-app.onrender.com/health
```
Expected: `{"status": "healthy", "model_loaded": true, ...}`

### 2. Model Info
```bash
curl https://your-app.onrender.com/model/info
```
Expected: Returns model metadata and performance metrics

### 3. Guest Prediction (No API Key)
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "interval": "1m"}'
```
Expected: Returns prediction (uses 1 of 3 free guest calls)

### 4. Create API Key
```bash
curl -X POST https://your-app.onrender.com/api-key/create \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "name": "Your Name",
    "quota_limit": 1000
  }'
```
Expected: Returns new API key

### 5. Authenticated Prediction
```bash
curl -X POST https://your-app.onrender.com/v1.1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"symbol": "BTCUSDT", "interval": "1m"}'
```
Expected: Returns enriched prediction with trends, tags, and suggestions

---

## Common Issues & Fixes

### ❌ MongoDB Connection Error
**Error**: `ServerSelectionTimeoutError: localhost:27017`
**Fix**: Add `MONGODB_URL` environment variable on Render

### ❌ Binance API Error
**Error**: `Invalid API-key, IP, or permissions`
**Fix**: 
1. Check `BINANCE_API_KEY` and `BINANCE_API_SECRET` are set
2. Verify API keys are valid on Binance
3. Enable "Spot & Margin Trading" permission (not required for data fetching, but good to have)

### ❌ Model Not Loading
**Error**: `Models not loaded. Please train models first.`
**Fix**: Ensure model files are in `data/models/` directory and committed with Git LFS

### ❌ 403 Free Trial Expired
**Error**: `Free trial expired. You've used all 3 free predictions`
**Fix**: 
- Create an API key using `/api-key/create` endpoint
- Or reset guest usage in MongoDB (for testing)

---

## MongoDB Atlas Configuration

### Connection String Format
```
mongodb+srv://<username>:<password>@cluster0.zdrqsde.mongodb.net/?retryWrites=true&w=majority
```

### URL Encoding for Passwords
If your password contains special characters, encode them:
- `@` → `%40`
- `#` → `%23`
- `!` → `%21`
- `$` → `%24`
- `%` → `%25`
- `&` → `%26`

Example:
```
Password: MyP@ss#123!
Encoded:  MyP%40ss%23123%21
```

### Database Collections
The API will automatically create these collections:
- `users` - API key holders
- `api_keys` - Active API keys (deprecated, now stored in users)
- `predictions` - Prediction history
- `guest_usage` - Guest IP tracking

---

## Render Build & Start Commands

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
python prediction_api.py
```

### Python Version
Ensure `runtime.txt` contains:
```
python-3.11.13
```

---

## Monitoring

### Logs
Check Render logs for:
- ✅ `✓ MongoDB connection established`
- ✅ `✓ Ensemble models loaded`
- ✅ `✓ API ready with models from:`
- ✅ `INFO: Uvicorn running on http://0.0.0.0:8000`

### Expected Startup Output
```
Starting Crypto Price Movement Prediction API
Connected to MongoDB: btc_prediction_api
✓ MongoDB connection established
Found model directory: data/models/BTCUSDT_1m_20251008_144639
✓ CatBoost model loaded
✓ Random Forest model loaded
✓ Logistic Regression model loaded (dict format)
✓ Meta-learner loaded
✓ Using stacking ensemble (base models → meta-learner)
✓ API ready with models from: data/models/BTCUSDT_1m_20251008_144639
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## API Endpoints Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | API status |
| `/health` | GET | No | Health check |
| `/model/info` | GET | No | Model metadata |
| `/predict` | POST | Optional | V1.0 predictions |
| `/v1.1/predict` | POST | Optional | V1.1 enriched predictions |
| `/api-key/create` | POST | No | Create new API key |
| `/api-key/revoke` | POST | No | Revoke API key by email |

---

## Success Criteria ✅

Your deployment is successful if:
- [x] Health endpoint returns 200 OK
- [x] MongoDB connection established (no localhost:27017 errors)
- [x] Models loaded successfully
- [x] Predictions return valid JSON with confidence scores
- [x] Guest usage tracking works (3 free calls)
- [x] API key authentication works
- [x] Streamlit frontend can connect and get predictions

---

## Next Steps

1. **Test from Streamlit**: Update your Streamlit app to use the Render URL
2. **Create Production API Keys**: Use `/api-key/create` for real users
3. **Monitor Usage**: Check MongoDB for prediction logs and usage stats
4. **Set Up Alerts**: Configure Render alerts for downtime
5. **Performance Tuning**: Monitor response times and optimize if needed

---

## Support

If issues persist:
1. Check Render logs for detailed error messages
2. Verify all environment variables are set correctly
3. Test MongoDB connection string locally first
4. Ensure Git LFS models are properly deployed
5. Contact Kevin Maglaqui: kevinroymaglaqui27@gmail.com

**Deployment Status**: 🟢 OPERATIONAL
**Last Updated**: October 8, 2025
