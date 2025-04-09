import asyncio
import faiss
import numpy as np
import os
from typing import List, Tuple

class AsyncWordDatabase:
    def __init__(self, index_path, words_list_path):
        self.index_path = index_path
        self.words_list_path = words_list_path
        self.index = None
        self.words = None
        self.load_lock = asyncio.Lock()

    async def load(self):
        async with self.load_lock:
            if self.index is not None and self.words is not None:
                return  # Already loaded

            loop = asyncio.get_running_loop()
            
            # Load index
            def load_index_sync():
                if not os.path.exists(self.index_path):
                    raise FileNotFoundError(f"Index file not found: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
            
            # Load words list
            def load_words_sync():
                if not os.path.exists(self.words_list_path):
                    raise FileNotFoundError(f"Words list file not found: {self.words_list_path}")
                with open(self.words_list_path, 'r', encoding='utf-8') as f:
                    self.words = [line.strip() for line in f]

            await loop.run_in_executor(None, load_index_sync)
            await loop.run_in_executor(None, load_words_sync)

            if self.index.ntotal != len(self.words):
                raise ValueError(
                    f"Number of vectors in index ({self.index.ntotal}) "
                    f"does not match number of words ({len(self.words)})"
                )

    def __len__(self):
        return len(self.words) if self.words else 0

    async def word_exists(self, word):
        await self.load()
        return word in self.words

    async def get_word_vector(self, word: str) -> np.ndarray:
        await self.load()
        try:
            index = self.words.index(word)
            return self.index.reconstruct(index)
        except ValueError:
            raise ValueError(f"Word '{word}' not found in database")

    async def get_similar_words(
        self,
        vector: np.ndarray,
        k: int = 5,
        min_distance: float = None,
        max_distance: float = None
    ) -> List[Tuple[str, float]]:
        """
        Поиск ближайших векторов в FAISS с фильтрацией по расстоянию.
        """
        await self.load()
        
        if min_distance is not None or max_distance is not None:
            # Реализация фильтрации по расстоянию "вручную"
            D, I = self.index.search(np.expand_dims(vector, axis=0), k=len(self))
            
            results = []
            for i, index in enumerate(I[0]):
                if index == -1:
                    continue  # No result
                
                distance = D[0][i]
                if (min_distance is None or distance >= min_distance) and \
                   (max_distance is None or distance <= max_distance):
                    
                    word = self.words[index]
                    results.append((word, float(distance)))
            
            # Возвращаем топ-k
            results.sort(key=lambda x: x[1])  # Сортируем по расстоянию
            return results[:k]
        else:
            # Используем стандартный поиск FAISS
            D, I = self.index.search(np.expand_dims(vector, axis=0), k=k)
            return [(self.words[index], float(distance)) for distance, index in zip(D[0], I[0]) if index != -1]

    async def get_all_word_vectors(self) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает все векторы слов из базы данных.
        
        Returns:
            List[Tuple[str, np.ndarray]]: Список кортежей, где первый элемент - слово, второй - его вектор.
        """
        await self.load()
        
        word_vectors = []
        for i, word in enumerate(self.words):
            vector = self.index.reconstruct(i)
            word_vectors.append((word, vector))
        
        return word_vectors
