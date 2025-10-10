import requests
import os

BASE = 'http://127.0.0.1:8000'

def test_health():
    r = requests.get(BASE + '/')
    print('GET / ->', r.status_code, r.text)

def test_db_health():
    r = requests.get(BASE + '/database/health')
    print('GET /database/health ->', r.status_code, r.text)

def test_model_info():
    r = requests.get(BASE + '/model/info')
    print('GET /model/info ->', r.status_code, r.text)

def test_transformer_endpoint():
    payload = {
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'use_live_data': False
    }
    r = requests.post(BASE + '/v2/transformermodel/keras', json=payload)
    print('POST /v2/transformermodel/keras ->', r.status_code, r.text)

def test_generate_api_key():
    headers = {'X-Admin-Secret': os.getenv('ADMIN_SECRET', 'test_admin_secret')}
    r = requests.post(BASE + '/api-keys/generate?name=test&email=test@example.com', headers=headers)
    print('POST /api-keys/generate ->', r.status_code, r.text)

if __name__ == '__main__':
    test_health()
    test_db_health()
    test_model_info()
    test_transformer_endpoint()
    test_generate_api_key()
