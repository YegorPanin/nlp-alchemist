#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message, BotCommand
from aiogram.utils.markdown import hbold
from aiogram.enums import ParseMode

from replicas import GREETINGS, COMMANDS, RESPONSES, ERRORS, SUCCESS
from alchemist import WordAlchemist
from database import AsyncWordDatabase

from leaderboard import MongoDBManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlchemyBot:
    def __init__(self, token):
        logger.info("Initializing AlchemyBot")
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()
        self.db = AsyncWordDatabase("word_embeddings.faiss", "words.list")
        self.alchemist = WordAlchemist(self.db)
        self.leaderboard = MongoDBManager("leaderboard")
        self.leaderboard.ensure_indexes()
        
        # Регистрация роутеров
        self.dp.include_router(self.setup_handlers())

    async def set_bot_commands(self):
        """Установка команд меню"""
        commands = [
            BotCommand(command="start", description="Начать работу"),
            BotCommand(command="help", description="Помощь"),
            BotCommand(command="commands", description="📜 Вспомнить формулы"),
            BotCommand(command="similar", description="🔍 Поиск похожих слов"),
            BotCommand(command="analogy", description="⚗ Решить аналогию"),
            BotCommand(command="mix", description="🔄 Смешать слова с операторами и множителями"),
            BotCommand(command="between", description="⇄ Найти промежуточные слова"),
            BotCommand(command="leaders", description="🏆 Таблица лучших алхимиков")
        ]
        await self.bot.set_my_commands(commands)

    async def check_similarity(self, user, word, similarity):
        """Проверяет сходство и обновляет таблицу лидеров"""
        if not await self.leaderboard.get_user(user.id):
            await self.leaderboard.create_user(user.id, user.first_name)
        
        if similarity > 0.8:  # Снизили порог для новых слов
            await self.leaderboard.increment_score(user.id, 1)
            await self.leaderboard.add_words(user.id, [word])
            return True
        return False

    def setup_handlers(self):
        router = Router()
        
        @router.message(Command("start"))
        async def start(message: Message):
            """Обработчик команды /start"""
            logger.info(f"Start command from user {message.from_user.id}")
            try:
                await message.answer(GREETINGS['start'].format(user=hbold(message.from_user.first_name)))
            except Exception as e:
                logger.error(f"Error in start command: {e}")
                await message.answer(ERRORS['default'])

        @router.message(Command("help"))
        async def help(message: Message):
            """Обработчик команды /help"""
            logger.info(f"Help command from user {message.from_user.id}")
            try:
                help_text = GREETINGS['help'] + "\n\n"
                for cmd, data in COMMANDS.items():
                    help_text += f"• {data['description']}\nПример: {data['example']}\n\n"
                help_text += f"Использование /mix:\n"
                help_text += "  /mix [множитель] слово [оператор] [множитель] слово ...\n"
                help_text += "  Пример: /mix 0.5 корова + 0.1 бык - 0.2 вымя\n\n"
                help_text += f"Использование /similar:\n"
                help_text += "  /similar слово [количество]\n"
                help_text += "  Пример: /similar кошка 10\n\n"
                await message.answer(help_text)
            except Exception as e:
                logger.error(f"Error in help command: {e}")
                await message.answer(ERRORS['default'])

        @router.message(Command("similar"))
        async def similar(message: Message):
            """Поиск семантически похожих слов"""
            try:
                args = message.text.split()[1:]
                if len(args) < 1:
                    raise ValueError
                
                word = args[0].strip()
                try:
                    count = int(args[1])
                    if count > 20:
                        count = 20
                    elif count < 1:
                        count = 5
                except (IndexError, ValueError):
                    count = 5
                
                logger.info(f"Similar words search for '{word}' by user {message.from_user.id}")
                results = await self.alchemist.find_similar_words(word, k=count+1)  # Запрашиваем на 1 больше
                # Фильтруем исходное слово из результатов
                results = [(w, s) for w, s in results if w.lower() != word.lower()][:count]
                logger.info(f"Found {len(results)} similar words for '{word}'")
                formatted = "\n".join(f"{i+1}. [{w}] ({s:.2f})" for i, (w, s) in enumerate(results))
                response = RESPONSES['similar_results'].format(
                    word=word, 
                    results=formatted
                )
                await message.answer(response)
                
                # Проверяем максимальное сходство и обновляем таблицу лидеров
                if results:
                    max_similarity = max(s for _, s in results)
                    if await self.check_similarity(message.from_user, word, max_similarity):
                        await message.answer(SUCCESS['discovery'])
                
            except (ValueError, IndexError):
                await message.answer(ERRORS['missing_args'])
            except Exception as e:
                logger.exception(f"Error in similar command: {e}", exc_info=True)
                if "Слово" in str(e):
                    word = str(e).split("'")[1]
                    await message.answer(
                        ERRORS['word_not_found'].format(word=word)
                    )
                else:
                    await message.answer(ERRORS['invalid_input'])

        @router.message(Command("analogy"))
        async def analogy(message: Message):
            """Решение аналогий"""
            try:
                args = message.text.split()[1:]
                if len(args) != 3:
                    raise ValueError
                    
                word_a, word_b, word_c = args
                logger.info(f"Analogy request: {word_a}:{word_b}::{word_c}:? by user {message.from_user.id}")
                results = await self.alchemist.word_analogy(word_a, word_b, word_c)
                logger.info(f"Found {len(results)} analogy results")
                formatted = "\n".join(f"- {w} ({s:.2f})" for w, s in results)
                
                response = RESPONSES['analogy_results'].format(
                    word_a=word_a,
                    word_b=word_b,
                    word_c=word_c,
                    results=formatted
                )
                await message.answer(response)
                
                # Проверяем максимальное сходство и обновляем таблицу лидеров
                if results:
                    max_similarity = max(s for _, s in results)
                    if await self.check_similarity(message.from_user, word_c, max_similarity):
                        await message.answer(SUCCESS['discovery'])
                
            except (ValueError, IndexError):
                await message.answer(ERRORS['missing_args'])
            except Exception as e:
                logger.exception(f"Error in analogy command: {e}", exc_info=True)
                if "Слово" in str(e):
                    word = str(e).split("'")[1]
                    await message.answer(
                        ERRORS['word_not_found'].format(word=word)
                    )
                else:
                    await message.answer(ERRORS['invalid_input'])

        @router.message(Command("mix"))
        async def mix(message: Message):
            """Смешивание семантики слов"""
            try:
                expression = message.text.split(maxsplit=1)[1].strip()
                if not expression:
                    raise ValueError

                # Разбираем выражение на слова, операторы и множители
                parts = expression.split()
                words = []
                operators = []
                multipliers = []
                current_word = ""
                current_multiplier = 1.0
                expect_word = True  # Ожидаем слово или множитель, а не оператор
                
                for i, part in enumerate(parts):
                    if part in ["+", "-"]:
                        # Проверяем, что перед оператором было слово
                        if not current_word.strip() and i > 0:
                            raise ValueError("Оператор без предшествующего слова")
                        
                        operators.append(part)
                        if current_word.strip():  # Добавляем слово только если оно не пустое
                            words.append(current_word.strip())
                            multipliers.append(current_multiplier)
                        current_word = ""
                        current_multiplier = 1.0
                        expect_word = True
                    else:
                        try:
                            # Попытка преобразовать часть в множитель
                            multiplier = float(part)
                            current_multiplier = multiplier
                        except ValueError:
                            # Если не удалось, добавляем к текущему слову
                            current_word += part + " "
                
                # Добавляем последнее слово, если оно есть
                if current_word.strip():
                    words.append(current_word.strip())
                    multipliers.append(current_multiplier)
                
                # Проверяем, что у нас есть хотя бы одно слово
                if not words:
                    raise ValueError("Не указаны слова для смешивания")
                
                # Проверяем соответствие количества слов, операторов и множителей
                if len(words) != len(multipliers):
                    raise ValueError("Несоответствие количества слов и множителей")
                
                if operators and len(operators) != len(words) - 1:
                    raise ValueError("Несоответствие количества слов и операторов")

                logger.info(f"Semantic mix request for expression: {expression} by user {message.from_user.id}")
                results = await self.alchemist.semantic_mix(words, operators, multipliers)
                logger.info(f"Semantic mix completed with {len(results)} results")
                formatted = "\n".join(f"- {w} ({s:.2f})" for w, s in results)

                response = RESPONSES['mix_results'].format(
                    words=", ".join(words),
                    results=formatted
                )
                await message.answer(response)
                
                # Проверяем максимальное сходство и обновляем таблицу лидеров
                if results:
                    max_similarity = max(s for _, s in results)
                    if await self.check_similarity(message.from_user, results[0][0], max_similarity):
                        await message.answer(SUCCESS['discovery'])

            except (ValueError, IndexError):
                await message.answer(ERRORS['missing_args'])
            except Exception as e:
                logger.exception(f"Error in mix command: {e}", exc_info=True)
                if "Слово" in str(e):
                    word = str(e).split("'")[1]
                    await message.answer(
                        ERRORS['word_not_found'].format(word=word)
                    )
                else:
                    await message.answer(ERRORS['invalid_input'])

        @router.message(Command("between"))
        async def between(message: Message):
            """Поиск промежуточных слов"""
            try:
                args = message.text.split()[1:]
                if len(args) != 2:
                    raise ValueError
                    
                word_a, word_b = args
                logger.info(f"Between words search: {word_a} and {word_b} by user {message.from_user.id}")
                results = await self.alchemist.closest_to_line(word_a, word_b)
                logger.info(f"Found {len(results)} between words results")
                formatted = "\n".join(f"- {w} ({s:.2f})" for w, s in results)
                
                response = RESPONSES['between_results'].format(
                    word_a=word_a,
                    word_b=word_b,
                    results=formatted
                )
                await message.answer(response)
                
                # Проверяем максимальное сходство и обновляем таблицу лидеров
                if results:
                    max_similarity = max(s for _, s in results)
                    if await self.check_similarity(message.from_user, results[0][0], max_similarity):
                        await message.answer(SUCCESS['discovery'])
                
            except (ValueError, IndexError):
                await message.answer(ERRORS['missing_args'])
            except Exception as e:
                logger.exception(f"Error in between command: {e}", exc_info=True)
                if "Слово" in str(e):
                    word = str(e).split("'")[1]
                    await message.answer(
                        ERRORS['word_not_found'].format(word=word)
                    )
                else:
                    await message.answer(ERRORS['invalid_input'])

        @router.message(Command("commands"))
        async def show_commands(message: Message):
            """Показать список всех команд"""
            logger.info(f"Commands list requested by user {message.from_user.id}")
            help_text = ""
            for cmd, data in COMMANDS.items():
                help_text += f"• {data['description']}\nПример: {data['example']}\n\n"
            await message.answer(help_text)

        @router.message(Command("leaders"))
        async def show_leaders(message: Message):
            """Показать таблицу лидеров"""
            try:
                logger.info(f"Leaderboard requested by user {message.from_user.id}")
                
                # Получаем топ-10
                leaders = await self.leaderboard.get_leaderboard(10)
                
                # Получаем позицию текущего пользователя
                user = await self.leaderboard.get_user(message.from_user.id)
                user_position = None
                if user:
                    all_users = await self.leaderboard.get_leaderboard(0)  # 0 = все пользователи
                    user_position = next((i+1 for i, u in enumerate(all_users) 
                                        if u['user_id'] == message.from_user.id), None)
                
                # Формируем сообщение
                leader_text = "🏆 *Топ алхимиков* 🏆\n\n"
                for i, leader in enumerate(leaders):
                    leader_text += f"{i+1}. {leader['user_name']} - {leader['score']} очков\n"
                
                if user_position:
                    leader_text += f"\nВаша позиция: {user_position}"
                else:
                    leader_text += "\nВы еще не участвуете в рейтинге"
                
                await message.answer(leader_text)
                
            except Exception as e:
                logger.error(f"Error in leaders command: {e}")
                await message.answer(ERRORS['default'])

        return router

    async def run(self):
        """Запуск бота"""
        logger.info("Starting bot polling")
        await self.dp.start_polling(self.bot)
        logger.info("Bot polling stopped")


async def main():
    import os
    from dotenv import load_dotenv
    
    # Загрузка переменных окружения из .env файла
    load_dotenv()
    
    # Получение токена из переменных окружения
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN не найден в переменных окружения")
        return
        
    bot = AlchemyBot(token)
    await bot.set_bot_commands()
    try:
        await bot.run()
    finally:
        await bot.bot.session.close()

if __name__ == '__main__':
    asyncio.run(main())
