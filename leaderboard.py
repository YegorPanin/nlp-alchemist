from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure, OperationFailure
from pymongo import DESCENDING
import asyncio
import logging
import functools
import os

def retry_operation(retries=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionFailure, OperationFailure) as e:
                    last_error = e
                    if attempt < retries - 1:
                        logging.warning(f"Database operation failed, attempt {attempt + 1} of {retries}: {e}")
                        await asyncio.sleep(delay * (attempt + 1))
                    continue
            logging.error(f"Database operation failed after {retries} attempts: {last_error}")
            raise last_error
        return wrapper
    return decorator

class MongoDBManager:
    def __init__(self, database_name: str, mongo_uri: str = None):
        try:
            # Use environment variable if no URI provided
            mongo_uri = mongo_uri or os.environ.get('MONGO_URI', "mongodb://localhost:27017")
            self.client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[database_name]
            self.users = self.db.users
            self._connected = False
        except Exception as e:
            logging.error(f"Failed to initialize MongoDB connection: {e}")
            raise

    async def connect(self):
        """Ensure connection to the database"""
        if not self._connected:
            try:
                await self.client.admin.command('ping')
                self._connected = True
                logging.info("Successfully connected to MongoDB")
            except Exception as e:
                logging.error(f"Failed to connect to MongoDB: {e}")
                raise

    async def ensure_connection(self):
        """Ensure we have a valid connection before operations"""
        if not self._connected:
            await self.connect()

    async def ensure_indexes(self):
        """Создаем уникальный индекс для user_id и индекс для сортировки по score"""
        await self.ensure_connection()
        await self.users.create_index([("user_id", 1)], unique=True)

    @retry_operation()
    async def create_user(self, user_id: int, user_name: str) -> bool:
        """Создает нового пользователя"""
        await self.ensure_connection()
        user_document = {
            "user_id": user_id,
            "user_name": user_name,
            "collected_words": [],
            "score": 0
        }
        try:
            await self.users.insert_one(user_document)
            return True
        except DuplicateKeyError:
            return False

    @retry_operation()
    async def get_user(self, user_id: int) -> dict:
        """Возвращает пользователя по ID"""
        await self.ensure_connection()
        return await self.users.find_one({"user_id": user_id})

    async def close(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client'):
            await self.client.close()
            self._connected = False

    @retry_operation()
    async def increment_score(self, user_id: int, points: int) -> bool:
        """Увеличивает счет пользователя"""
        await self.ensure_connection()
        result = await self.users.update_one(
            {"user_id": user_id},
            {"$inc": {"score": points}}
        )
        return result.modified_count > 0

    @retry_operation()
    async def add_words(self, user_id: int, words: list) -> bool:
        """Добавляет слова в коллекцию пользователя"""
        if not words:
            return False
            
        await self.ensure_connection()
        result = await self.users.update_one(
            {"user_id": user_id},
            {"$addToSet": {"collected_words": {"$each": words}}}
        )
        return result.modified_count > 0

    async def get_leaderboard(self, limit: int = 10) -> list:
        """Возвращает отсортированный список лидеров"""
        leaderboard = []
        async for user in self.users.find().sort("score", DESCENDING).limit(limit):
            leaderboard.append(user)
        return leaderboard

    async def close(self):
        """Закрывает соединение с базой данных"""
        self.client.close()
