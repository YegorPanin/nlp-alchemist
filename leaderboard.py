from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from pymongo import DESCENDING
import asyncio

class MongoDBManager:
    def __init__(self, database_name: str, mongo_uri: str = "mongodb://localhost:27017"):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]
        self.users = self.db.users

    async def ensure_indexes(self):
        # Создаем уникальный индекс для user_id и индекс для сортировки по score
        await self.users.create_index([("user_id", 1)], unique=True)

    async def create_user(self, user_id: int, user_name: str) -> bool:
        """Создает нового пользователя"""
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

    async def get_user(self, user_id: int) -> dict:
        """Возвращает пользователя по ID"""
        return await self.users.find_one({"user_id": user_id})

    async def increment_score(self, user_id: int, points: int) -> bool:
        """Увеличивает счет пользователя"""
        result = await self.users.update_one(
            {"user_id": user_id},
            {"$inc": {"score": points}}
        )
        return result.modified_count > 0

    async def add_words(self, user_id: int, words: list) -> bool:
        """Добавляет слова в коллекцию пользователя"""
        if not words:
            return False
            
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
