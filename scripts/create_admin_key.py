import asyncio
import os
import uuid
from database import db

async def create_key(name, email):
    await db.connect()
    # If user exists, delete it to avoid duplicate key error (dev convenience)
    existing = await db.db.users.find_one({'email': email})
    if existing:
        print('Existing user found; removing for fresh key')
        await db.db.users.delete_one({'email': email})

    user = await db.create_user(email=email, name=name, quota_limit=1000)
    print('Created user:')
    print('email:', email)
    print('name:', name)
    print('api_key:', user['api_key'])
    await db.disconnect()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='admin')
    parser.add_argument('--email', default=f'admin+{uuid.uuid4().hex[:8]}@example.com')
    args = parser.parse_args()
    asyncio.run(create_key(args.name, args.email))
