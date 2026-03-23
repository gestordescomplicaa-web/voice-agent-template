import os
import json
import asyncpg
from datetime import datetime, timezone
from dotenv import load_dotenv

from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, silero, openai
from livekit.plugins.fishaudio import TTS as FishTTS

load_dotenv()


# =========================
# HISTÓRICO (OPCIONAL)
# =========================

async def buscar_historico(numero: str):
    if not os.getenv("POSTGRES_URL"):
        return ""

    try:
        conn = await asyncpg.connect(os.getenv("POSTGRES_URL"))
        rows = await conn.fetch("""
            SELECT message FROM n8n_chatwoot
            WHERE session_id = $1
            ORDER BY id DESC LIMIT 10
        """, numero)

        await conn.close()

        mensagens = []
        for row in reversed(rows):
            msg = row["message"]
            if isinstance(msg, str):
                msg = json.loads(msg)

            tipo = msg.get("type")
            conteudo = msg.get("content", "")

            if tipo == "human":
                mensagens.append(f"Cliente: {conteudo}")
            elif tipo == "ai":
                mensagens.append(f"Assistente: {conteudo}")

        return "\n".join(mensagens)

    except:
        return ""


# =========================
# AGENTE
# =========================

class Assistente(Agent):
    def __init__(self, numero, historico):
        instrucao_base = os.getenv("AGENT_PROMPT")

        historico_txt = (
            f"\n\nHistórico:\n{historico}\n"
            "Use isso apenas como contexto."
        ) if historico else ""

        super().__init__(
            instructions=(
                f"{instrucao_base} "
                "Seja breve e natural, como uma conversa por voz."
                f"{historico_txt}"
            )
        )


# =========================
# ENTRYPOINT
# =========================

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    numero_cliente = "desconhecido"
    room_name = ctx.room.name or ""

    for parte in room_name.split("_"):
        if parte.isdigit():
            numero_cliente = parte
            break

    historico = await buscar_historico(numero_cliente)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="pt-BR"),
        llm=openai.LLM(model=os.getenv("OPENAI_MODEL")),
        tts=FishTTS(reference_id=os.getenv("FISH_REFERENCE_ID")),
    )

    agente = Assistente(numero_cliente, historico)

    await session.start(room=ctx.room, agent=agente)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
