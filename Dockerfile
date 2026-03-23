FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "livekit-agents[openai,deepgram,silero,fishaudio]>=1.0" \
    livekit-api \
    python-dotenv \
    httpx \
    asyncpg

COPY agente.py .

CMD ["python", "agente.py", "start"]
