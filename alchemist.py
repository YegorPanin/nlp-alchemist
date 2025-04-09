import asyncio
import numpy as np
from typing import List, Tuple, Optional

class WordAlchemist:
    def __init__(self, db):
        self.db = db
    
    async def find_similar_words(
        self,
        word: str,
        k: int = 5,
        min_similarity: Optional[float] = None,
        max_similarity: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Находит семантически похожие слова с возможностью фильтрации по схожести

        Args:
            word: Исходное слово для поиска
            k: Количество возвращаемых результатов
            min_similarity: Минимальная степень схожести (0-1)
            max_similarity: Максимальная степень схожести (0-1)

        Returns:
            Список кортежей (слово, степень схожести)
        """
        if not await self.db.word_exists(word):
            raise ValueError(f"Слово '{word}' отсутствует в базе")

        vector = await self.db.get_word_vector(word)

        # Преобразуем параметры схожести в расстояния FAISS
        min_dist = 1 - max_similarity if max_similarity is not None else None
        max_dist = 1 - min_similarity if min_similarity is not None else None

        return await self.db.get_similar_words(
            vector,
            k=k,
            min_distance=min_dist,
            max_distance=max_dist
        )
    
    async def word_analogy(
        self, 
        word_a: str, 
        word_b: str, 
        word_c: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Решает аналогии вида: a относится к b, как c относится к ?
        Использует метод косинусного сходства между направлениями векторов.
        
        Пример:
            король относится к мужчина, как королева относится к ?
            word_analogy("король", "мужчина", "королева")
        """
        for word in [word_a, word_b, word_c]:
            if not await self.db.word_exists(word):
                raise ValueError(f"Слово '{word}' отсутствует в базе")
                
        vec_a = await self.db.get_word_vector(word_a)
        vec_b = await self.db.get_word_vector(word_b)
        vec_c = await self.db.get_word_vector(word_c)
        
        # Вычисляем направление аналогии: b - a
        direction = vec_b - vec_a
        direction /= np.linalg.norm(direction)  # Нормализуем
        
        # Для каждого слова в базе вычисляем косинусное сходство направлений
        results = []
        for word_d, vec_d in await self.db.get_all_word_vectors():
            if word_d in [word_a, word_b, word_c]:
                continue
                
            # Направление от c к d
            direction_d = vec_d - vec_c
            direction_d /= np.linalg.norm(direction_d)
            
            # Косинусное сходство между направлениями
            similarity = np.dot(direction, direction_d)
            results.append((word_d, float(similarity)))
        
        # Сортируем по убыванию сходства и возвращаем топ-k
        results.sort(key=lambda x: -x[1])
        return results[:k]
    
    async def semantic_mix(
        self,
        words: List[str],
        operators: Optional[List[str]] = None,
        multipliers: Optional[List[float]] = None,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Смешивает семантику нескольких слов с использованием нормализованного среднего
        
        Args:
            words: Список слов для смешивания
            operators: Список операторов (+, -). Если None, то все слова складываются.
            multipliers: Список множителей для слов. Если None, то все слова имеют множитель 1.
            k: Количество возвращаемых результатов
        """
        if not words:
            raise ValueError("Список слов не может быть пустым")

        if operators and len(operators) != len(words) - 1:
            raise ValueError("Количество операторов должно быть на один меньше количества слов")

        if multipliers and len(multipliers) != len(words):
            raise ValueError("Количество множителей должно совпадать с количеством слов")

        # Получаем векторы всех слов
        vectors = []
        for word in words:
            word = word.lower()
            if not await self.db.word_exists(word):
                raise ValueError(f"Слово '{word}' отсутствует в базе")
            vectors.append(await self.db.get_word_vector(word))

        # Если множители не указаны, используем 1.0 для всех слов
        if not multipliers:
            multipliers = [1.0] * len(words)
        
        # Нормализуем каждый вектор
        normalized_vectors = []
        for vec in vectors:
            norm = np.linalg.norm(vec)
            if norm > 0:  # Избегаем деления на ноль
                normalized_vectors.append(vec / norm)
            else:
                normalized_vectors.append(vec)
        
        # Инициализируем смешанный вектор с учетом операторов
        if operators:
            # Начинаем с первого вектора
            mixed_vector = normalized_vectors[0] * multipliers[0]
            
            # Применяем операторы
            for i, operator in enumerate(operators):
                if operator == "+":
                    mixed_vector += normalized_vectors[i + 1] * multipliers[i + 1]
                elif operator == "-":
                    mixed_vector -= normalized_vectors[i + 1] * multipliers[i + 1]
                else:
                    raise ValueError(f"Неизвестный оператор: {operator}")
        else:
            # Если операторы не указаны, вычисляем взвешенное среднее
            mixed_vector = np.zeros_like(normalized_vectors[0])
            total_weight = 0.0
            
            for i, vec in enumerate(normalized_vectors):
                weight = multipliers[i]
                mixed_vector += vec * weight
                total_weight += weight
            
            # Делим на сумму весов для получения среднего
            if total_weight > 0:
                mixed_vector /= total_weight
        
        # Нормализуем результирующий вектор
        norm = np.linalg.norm(mixed_vector)
        if norm > 0:
            mixed_vector = mixed_vector / norm

        # Ищем ближайшие слова к полученному вектору
        return await self.db.get_similar_words(mixed_vector, k=k)
    
    async def closest_to_line(
        self, 
        word_a: str, 
        word_b: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Находит слова, ближайшие к линии между двумя точками
        
        Полезно для поиска слов с промежуточными значениями признаков
        """
        if not await self.db.word_exists(word_a):
            raise ValueError(f"Слово '{word_a}' отсутствует в базе")
        if not await self.db.word_exists(word_b):
            raise ValueError(f"Слово '{word_b}' отсутствует в базе")
            
        vec_a = await self.db.get_word_vector(word_a)
        vec_b = await self.db.get_word_vector(word_b)
        
        # Вектор направления линии
        direction = vec_b - vec_a
        direction /= np.linalg.norm(direction)
        
        # Функция для проекции вектора на линию
        def project_on_line(vec):
            return vec_a + np.dot(vec - vec_a, direction) * direction
            
        # Поиск ближайших к проекции
        word_vectors = await self.db.get_all_word_vectors()
        results = []
        
        for word, vec in word_vectors:
            if word in [word_a, word_b]:
                continue
                
            projection = project_on_line(vec)
            distance = np.linalg.norm(vec - projection)
            results.append((word, float(distance)))
        
        # Сортируем по расстоянию и возвращаем топ-k
        results.sort(key=lambda x: x[1])
        return results[:k]
