"""
Create a test user with API key
"""
import asyncio
import sys
sys.path.insert(0, '.')
from database import Database

async def create_test_user():
    db = Database()
    await db.connect()
    
    print("Creating test user...")
    
    # Create user
    api_key = await db.create_user(
        email="test@example.com",
        name="Test User",
        quota_limit=1000  # High quota for testing
    )
    
    print(f"\n✅ Test user created!")
    print(f"   Email: test@example.com")
    print(f"   API Key: {api_key}")
    print(f"   Daily Quota: 1000 requests")
    print(f"\nUse this API key in your Streamlit app or tests.")
    print(f"Add it to request headers: Authorization: Bearer {api_key}")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(create_test_user())
