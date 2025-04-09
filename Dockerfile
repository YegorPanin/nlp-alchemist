FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY word_embeddings.faiss .
COPY words.list .

CMD ["python", "bot.py"]