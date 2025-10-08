import os
from datetime import datetime, timezone
from typing import Optional, Dict, List
import uuid
import bcrypt
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'btc_prediction_api')

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        
    async def connect(self):
        try:
            # Simplified connection - let pymongo handle SSL automatically with mongodb+srv://
            # The SRV connection string already implies TLS/SSL
            connection_params = {
                'serverSelectionTimeoutMS': 30000,
                'connectTimeoutMS': 30000,
                'socketTimeoutMS': 30000,
                'retryWrites': True,
                'w': 'majority',
            }
            
            self.client = AsyncIOMotorClient(MONGODB_URL, **connection_params)
            self.db = self.client[DATABASE_NAME]
            
            # Test the connection
            await self.client.admin.command('ping')
            
            await self._create_indexes()
            print(f'Connected to MongoDB: {DATABASE_NAME}')
        except Exception as e:
            print(f'MongoDB connection failed: {e}')
            raise
    
    async def disconnect(self):
        if self.client:
            self.client.close()
    
    async def _create_indexes(self):
        await self.db.users.create_index([('email', ASCENDING)], unique=True)
        await self.db.users.create_index([('api_key_hash', ASCENDING)], unique=True, sparse=True)
        await self.db.predictions.create_index([('user_id', ASCENDING)])
        await self.db.predictions.create_index([('timestamp', DESCENDING)])
        await self.db.guest_usage.create_index([('ip_address', ASCENDING)], unique=True)
    
    async def create_user(self, email: str, name: str, quota_limit: int = 1000):
        api_key = f'btc_{uuid.uuid4().hex}'
        api_key_hash = bcrypt.hashpw(api_key.encode(), bcrypt.gensalt()).decode()
        user = {
            '_id': str(uuid.uuid4()),
            'email': email,
            'name': name,
            'api_key_hash': api_key_hash,
            'created_at': datetime.now(timezone.utc),
            'requests_today': 0,
            'quota_limit': quota_limit,
            'last_quota_reset': datetime.now(timezone.utc),
            'is_active': True
        }
        await self.db.users.insert_one(user)
        user['api_key'] = api_key
        return user
    
    async def get_user_by_api_key(self, api_key: str):
        users = await self.db.users.find({'is_active': True}).to_list(length=None)
        for user in users:
            if bcrypt.checkpw(api_key.encode(), user['api_key_hash'].encode()):
                return user
        return None
    
    async def increment_user_requests(self, user_id: str):
        user = await self.db.users.find_one({'_id': user_id})
        if not user:
            return False
        now = datetime.now(timezone.utc)
        last_reset = user.get('last_quota_reset')
        # Handle both timezone-aware and naive datetimes
        if last_reset and not last_reset.tzinfo:
            last_reset = last_reset.replace(tzinfo=timezone.utc)
        if not last_reset:
            last_reset = now
        if (now - last_reset).days >= 1:
            await self.db.users.update_one(
                {'_id': user_id},
                {'$set': {'requests_today': 1, 'last_quota_reset': now}}
            )
            return True
        if user['requests_today'] >= user['quota_limit']:
            return False
        await self.db.users.update_one({'_id': user_id}, {'$inc': {'requests_today': 1}})
        return True
    
    async def get_user_usage(self, user_id: str):
        user = await self.db.users.find_one({'_id': user_id})
        if not user:
            return {}
        return {
            'email': user['email'],
            'name': user['name'],
            'requests_today': user['requests_today'],
            'quota_limit': user['quota_limit'],
            'requests_remaining': user['quota_limit'] - user['requests_today']
        }
    
    async def get_guest_usage(self, ip_address: str):
        guest = await self.db.guest_usage.find_one({'ip_address': ip_address})
        if not guest:
            guest = {
                'ip_address': ip_address,
                'calls_used': 0,
                'calls_limit': 3,
                'first_call': datetime.now(timezone.utc)
            }
            await self.db.guest_usage.insert_one(guest)
        return guest
    
    async def increment_guest_usage(self, ip_address: str):
        guest = await self.get_guest_usage(ip_address)
        if guest['calls_used'] >= guest['calls_limit']:
            return False
        await self.db.guest_usage.update_one(
            {'ip_address': ip_address},
            {'$inc': {'calls_used': 1}}
        )
        return True
    
    async def get_user_predictions(self, user_id: str, limit: int = 100):
        predictions = await self.db.predictions.find(
            {'user_id': user_id}
        ).sort('timestamp', DESCENDING).limit(limit).to_list(length=limit)
        return predictions
    
    async def get_prediction_stats(self, user_id: Optional[str] = None):
        query = {'user_id': user_id} if user_id else {}
        total = await self.db.predictions.count_documents(query)
        pipeline = [
            {'$match': query},
            {'$group': {
                '_id': None,
                'avg_confidence': {'$avg': '$confidence'},
                'avg_latency': {'$avg': '$latency_ms'}
            }}
        ]
        result = await self.db.predictions.aggregate(pipeline).to_list(length=1)
        return {
            'total_predictions': total,
            'avg_confidence': result[0]['avg_confidence'] if result else 0,
            'avg_latency_ms': result[0]['avg_latency'] if result else 0
        }
    
    async def register_model_version(self, version: str, notes: List[str], metrics: Dict):
        model_record = {
            'version': version,
            'release_date': datetime.now(timezone.utc),
            'notes': notes,
            'metrics': metrics,
            'is_active': True
        }
        await self.db.models.insert_one(model_record)
        print(f'Model version {version} registered')
    
    async def get_latest_model_version(self):
        return await self.db.models.find_one(
            {'is_active': True},
            sort=[('release_date', DESCENDING)]
        )
    
    async def revoke_api_key(self, user_id: str):
        result = await self.db.users.update_one(
            {'_id': user_id},
            {'$set': {'is_active': False}}
        )
        return result.modified_count > 0
    
    async def log_prediction(self, user_id, ip_address, symbol, interval, prediction, confidence, model_version, latency_ms, result):
        record = {
            '_id': str(uuid.uuid4()),
            'user_id': user_id or 'guest',
            'ip_address': ip_address,
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'interval': interval,
            'prediction': prediction,
            'confidence': confidence,
            'model_version': model_version,
            'latency_ms': latency_ms,
            'result': result
        }
        await self.db.predictions.insert_one(record)
        return record['_id']

db = Database()
