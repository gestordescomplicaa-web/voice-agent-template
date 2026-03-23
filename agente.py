import os
import json
import logging
from datetime import datetime, timezone
 
import httpx
from dotenv import load_dotenv
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, RunContext, cli, function_tool
from livekit.plugins import deepgram, silero, openai
from livekit.plugins.fishaudio import TTS as FishTTS
 
load_dotenv()
 
log = logging.getLogger("voice-agent")
log.setLevel(logging.INFO)
 
 
# -------------------------------------------
# helpers
# -------------------------------------------
 
def env(key: str, default: str = "") -> str:
    return (os.getenv(key) or "").strip() or default
 
 
def env_ok(key: str) -> bool:
    return bool(env(key))
 
 
async def post_webhook(url: str, payload: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            try:
                return resp.json()
            except Exception:
                return {"status": resp.status_code, "body": resp.text}
    except httpx.TimeoutException:
        log.warning("Webhook timeout: %s", url)
        return {"erro": "timeout"}
    except Exception as exc:
        log.error("Webhook falhou (%s): %s", url, exc)
        return {"erro": str(exc)}
 
 
# -------------------------------------------
# carregar tools do .env
# -------------------------------------------
 
def carregar_tools_env() -> list[dict]:
    tools = []
    i = 1
    while True:
        name = env(f"TOOL_{i}_NAME")
        url = env(f"TOOL_{i}_URL")
        desc = env(f"TOOL_{i}_DESC")
        campos_raw = env(f"TOOL_{i}_CAMPOS")
 
        if not name and not url:
            break
 
        if name and url and desc:
            campos = [c.strip() for c in campos_raw.split(",") if c.strip()] if campos_raw else []
            tools.append({
                "name": name,
                "url": url,
                "desc": desc,
                "campos": campos,
            })
            log.info("Tool carregada: %s -> %s (campos: %s)", name, url, campos)
        else:
            log.warning("TOOL_%d incompleta (falta NAME, URL ou DESC) -- ignorada", i)
 
        i += 1
 
    return tools
 
 
def criar_function_tool(tool_cfg: dict):
    name = tool_cfg["name"]
    url = tool_cfg["url"]
    desc = tool_cfg["desc"]
    campos = tool_cfg["campos"]
 
    properties = {}
    for campo in campos:
        properties[campo] = {
            "type": "string",
            "description": campo.replace("_", " "),
        }
 
    raw_schema = {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": [campos[0]] if campos else [],
        },
    }
 
    async def handler(raw_arguments: dict, ctx: RunContext) -> str:
        numero = "desconhecido"
        if hasattr(ctx, "agent") and hasattr(ctx.agent, "_numero"):
            numero = ctx.agent._numero
 
        payload = {
            "numero": numero,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for campo in campos:
            payload[campo] = raw_arguments.get(campo, "")
 
        log.info("Tool %s chamada -> %s | payload: %s", name, url, json.dumps(payload, ensure_ascii=False))
        resultado = await post_webhook(url, payload)
 
        if "erro" in resultado:
            return f"Erro ao executar {name}: {resultado['erro']}"
        return json.dumps(resultado, ensure_ascii=False) if isinstance(resultado, dict) else str(resultado)
 
    return function_tool(handler, raw_schema=raw_schema)
 
 
# -------------------------------------------
# historico (opcional)
# -------------------------------------------
 
async def buscar_historico(numero: str) -> str:
    if not env_ok("POSTGRES_URL"):
        return ""
    try:
        import asyncpg
        conn = await asyncpg.connect(env("POSTGRES_URL"))
        rows = await conn.fetch(
            """
            SELECT message FROM n8n_chatwoot
            WHERE session_id = $1
            ORDER BY id DESC LIMIT 10
            """,
            numero,
        )
        await conn.close()
 
        linhas: list[str] = []
        for row in reversed(rows):
            msg = row["message"]
            if isinstance(msg, str):
                msg = json.loads(msg)
            tipo = msg.get("type")
            conteudo = msg.get("content", "")
            if tipo == "human":
                linhas.append(f"Cliente: {conteudo}")
            elif tipo == "ai":
                linhas.append(f"Assistente: {conteudo}")
        return "\n".join(linhas)
    except ImportError:
        log.warning("asyncpg nao instalado -- historico desativado")
        return ""
    except Exception as exc:
        log.warning("Erro ao buscar historico: %s", exc)
        return ""
 
 
# -------------------------------------------
# extrair numero do cliente
# -------------------------------------------
 
def extrair_numero(room_name: str) -> str:
    for parte in room_name.split("_"):
        if parte.isdigit() and len(parte) >= 8:
            return parte
    return "desconhecido"
 
 
# -------------------------------------------
# carregar tools uma vez na inicializacao
# -------------------------------------------
 
TOOLS_CONFIG = carregar_tools_env()
DYNAMIC_TOOLS = [criar_function_tool(cfg) for cfg in TOOLS_CONFIG]
 
 
# -------------------------------------------
# agente
# -------------------------------------------
 
class Assistente(Agent):
    def __init__(self, numero: str, historico: str):
        self._numero = numero
 
        prompt_base = env("AGENT_PROMPT", "Voce e um atendente virtual simpatico e objetivo.")
        partes = [
            prompt_base,
            "Seja breve e natural, como uma conversa por voz.",
            "Responda sempre em portugues brasileiro.",
        ]
 
        if TOOLS_CONFIG:
            partes.append("\nVoce tem acesso as seguintes ferramentas:")
            for cfg in TOOLS_CONFIG:
                partes.append(f"  - {cfg['name']}: {cfg['desc']}")
            partes.append("Use-as quando fizer sentido para atender o cliente.")
 
        if historico:
            partes.append(
                f"\n--- Historico recente ---\n{historico}\n"
                "Use isso apenas como contexto, nao repita."
            )
 
        super().__init__(
            instructions="\n".join(partes),
            tools=DYNAMIC_TOOLS,
        )
 
 
# -------------------------------------------
# entrypoint
# -------------------------------------------
 
async def entrypoint(ctx: JobContext):
    await ctx.connect()
 
    numero = extrair_numero(ctx.room.name or "")
    log.info("Sala: %s | Cliente: %s", ctx.room.name, numero)
    log.info("Tools ativas: %d", len(DYNAMIC_TOOLS))
 
    historico = await buscar_historico(numero)
 
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="pt-BR"),
        llm=openai.LLM(model=env("OPENAI_MODEL", "gpt-4.1-mini")),
        tts=FishTTS(reference_id=env("FISH_REFERENCE_ID")),
    )
 
    agente = Assistente(numero, historico)
    await session.start(room=ctx.room, agent=agente)
 
    saudacao = env("SAUDACAO")
    if saudacao:
        await session.say(saudacao)
 
 
# -------------------------------------------
# run
# -------------------------------------------
 
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
