FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "livekit-agents[openai,deepgram,silero,fishaudio]>=1.0" \
    python-dotenv

COPY agente.py .

CMD ["python", "agente.py", "start"]
