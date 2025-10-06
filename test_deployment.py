#!/usr/bin/env python3
"""
Test deployment script
Tests the API endpoints locally to verify deployment readiness
"""

import requests
import json
import time
import sys

def test_api(base_url="http://localhost:8000"):
    """Test API endpoints"""
    
    print(f"Testing API at {base_url}")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Response: {json.dumps(health_data, indent=2)}")
            models_loaded = health_data.get('model_loaded', False)
            print(f"   Models loaded: {models_loaded}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test 2: Root endpoint
    print("2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test 3: Model info
    print("3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            info_data = response.json()
            print(f"   Response: {json.dumps(info_data, indent=2)}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Test 4: Prediction (only if models are loaded)
    print("4. Testing prediction endpoint...")
    try:
        # First check if models are loaded
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200 and health_response.json().get('model_loaded', False):
            
            prediction_request = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "use_live_data": True
            }
            
            response = requests.post(
                f"{base_url}/predict", 
                json=prediction_request,
                timeout=30
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                pred_data = response.json()
                print(f"   Prediction: {pred_data.get('prediction_label', 'N/A')}")
                print(f"   Confidence: {pred_data.get('confidence', 'N/A'):.3f}")
                print(f"   Current price: ${pred_data.get('current_price', 'N/A')}")
            else:
                print(f"   Error: {response.text}")
        else:
            print("   Skipped: Models not loaded")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    print("=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    # Allow custom URL
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(url)