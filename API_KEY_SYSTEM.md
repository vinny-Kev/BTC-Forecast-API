# 🔐 API Key System & Rate Limiting

## Overview

The BTC Forecast API uses a dual-access system:
- **Guest Access**: 3 free predictions per IP address (no API key needed)
- **API Key Access**: 3 predictions per minute with an API key

## 🎯 For Users

### Guest Access (Free Trial)

No API key needed! Just make a request:

```bash
curl -X POST https://btc-forecast-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "interval": "1m"}'
```

**Limits:**
- ✅ 3 free predictions per IP address
- ❌ After 3 calls, you'll need an API key

### With API Key

```bash
curl -X POST https://btc-forecast-api.onrender.com/predict \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "interval": "1m"}'
```

**Benefits:**
- ✅ 3 predictions per minute (not lifetime)
- ✅ Continuous access
- ✅ Usage tracking

### Getting an API Key

**Contact:** Kevin Maglaqui  
**Email:** kevinroymaglaqui27@gmail.com

Include:
- Your name
- Your email
- Use case

## 🛠️ For Admins

### Generate API Key

```bash
curl -X POST "http://localhost:8000/api-keys/generate?name=User%20Name&email=user@email.com" \
  -H "X-Admin-Secret: YOUR_ADMIN_SECRET"
```

**Response:**
```json
{
  "success": true,
  "api_key": "btc_xxxxx...",
  "name": "User Name",
  "email": "user@email.com",
  "rate_limit": "3 predictions per minute"
}
```

### Check Usage

```bash
curl -X GET http://localhost:8000/api-keys/usage \
  -H "Authorization: Bearer API_KEY"
```

### Revoke API Key

```bash
curl -X DELETE http://localhost:8000/api-keys/revoke?api_key_to_revoke=btc_xxxxx \
  -H "X-Admin-Secret: YOUR_ADMIN_SECRET"
```

## 📊 Rate Limits

| Access Type | Limit | Window | Lifetime |
|------------|-------|--------|----------|
| **Guest** | 3 calls | Lifetime | Yes |
| **API Key** | 3 calls | 1 minute | No |

## 🔒 Security

### Set Admin Secret

**Production (Render):**
1. Go to Render Dashboard
2. Add environment variable: `ADMIN_SECRET=your_secure_secret`
3. Save and redeploy

**Local (.env file):**
```env
ADMIN_SECRET=your_local_secret
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
```

## 📝 API Endpoints

### Public Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Health check |
| `/health` | GET | No | Health check |
| `/predict` | POST | Optional* | Get predictions |
| `/model/info` | GET | No | Model information |

*Optional = Works without key (3 free calls), better with key

### Admin Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api-keys/generate` | POST | Admin | Generate new API key |
| `/api-keys/usage` | GET | API Key | Check usage |
| `/api-keys/revoke` | DELETE | Admin | Revoke API key |

## 🧪 Testing

Run the test suite:

```bash
# Test guest access only
python test_api_keys.py

# Test everything (with admin secret)
python test_api_keys.py YOUR_ADMIN_SECRET
```

## 💾 Data Storage

- **API keys**: Stored in `api_keys.json`
- **Guest usage**: Tracked by IP address in same file
- **Format**:
```json
{
  "api_keys": {
    "btc_xxxxx": {
      "name": "User Name",
      "email": "user@email.com",
      "created_at": "2025-10-06T...",
      "calls_made": 0
    }
  },
  "guest_usage": {
    "192.168.1.1": 3
  }
}
```

## 🚀 Deployment

The system automatically loads API keys on startup. Make sure to:

1. ✅ Set `ADMIN_SECRET` environment variable on Render
2. ✅ Keep `api_keys.json` in `.gitignore` (it's persistent on Render)
3. ✅ Monitor guest usage to prevent abuse

## 📞 Support

Trial limit reached? Contact Kevin Maglaqui for an API key!

**Email:** kevinroymaglaqui27@gmail.com  
**Message:** "Hi, I've used my 3 free predictions and would like an API key to continue using the BTC Forecast API."
