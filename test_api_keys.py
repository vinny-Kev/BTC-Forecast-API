#!/usr/bin/env python3
"""
Test API Key System and Rate Limiting
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_guest_access():
    """Test guest access (3 free calls)"""
    print("\n" + "="*60)
    print("Testing Guest Access (3 Free Calls)")
    print("="*60)
    
    for i in range(5):  # Try 5 calls to test the limit
        print(f"\n📞 Guest Call {i+1}:")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "use_live_data": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success: {data['prediction_label']}")
                print(f"   Confidence: {data['confidence']:.2%}")
            elif response.status_code == 403:
                error = response.json()
                print(f"   🚫 {error['detail']}")
                break
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        time.sleep(1)


def test_with_api_key(api_key):
    """Test with API key (3 calls per minute)"""
    print("\n" + "="*60)
    print("Testing With API Key")
    print("="*60)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test usage endpoint
    print("\n📊 Checking usage:")
    try:
        response = requests.get(f"{BASE_URL}/api-keys/usage", headers=headers)
        if response.status_code == 200:
            usage = response.json()
            print(f"   User: {usage.get('name', 'N/A')}")
            print(f"   Rate Limit: {usage.get('rate_limit', 'N/A')}")
            print(f"   Calls Remaining: {usage.get('calls_remaining', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test predictions
    print("\n📞 Making predictions:")
    for i in range(4):  # Try 4 calls to test rate limit
        print(f"\n   Call {i+1}:")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "use_live_data": True
                },
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"      ✅ {data['prediction_label']}")
            elif response.status_code == 429:
                error = response.json()
                print(f"      ⏱️ Rate limit: {error['detail']}")
                break
            else:
                print(f"      ❌ Error {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Exception: {e}")
        
        time.sleep(0.5)


def generate_api_key(admin_secret, name="Test User", email="test@example.com"):
    """Generate a new API key"""
    print("\n" + "="*60)
    print("Generating API Key")
    print("="*60)
    
    headers = {
        "X-Admin-Secret": admin_secret,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api-keys/generate",
            params={"name": name, "email": email},
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ API Key Generated!")
            print(f"   Name: {data['name']}")
            print(f"   Email: {data['email']}")
            print(f"   API Key: {data['api_key']}")
            print(f"   Rate Limit: {data['rate_limit']}")
            return data['api_key']
        else:
            print(f"❌ Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    print("\n🔐 API Key System Test Suite")
    print("=" * 60)
    
    # Test guest access
    test_guest_access()
    
    # Optionally test with admin secret
    if len(sys.argv) > 1:
        admin_secret = sys.argv[1]
        api_key = generate_api_key(admin_secret, "Kevin Maglaqui", "kevinroymaglaqui27@gmail.com")
        
        if api_key:
            print(f"\n💾 Save this API key: {api_key}")
            test_with_api_key(api_key)
    else:
        print("\n💡 To test API key generation:")
        print(f"   python {sys.argv[0]} YOUR_ADMIN_SECRET")
    
    print("\n✅ Tests completed!")
