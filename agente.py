import os
import json
import logging
import asyncio
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, RunContext, cli, function_tool
from livekit.plugins import deepgram, silero, openai
from livekit.plugins.fishaudio import TTS as FishTTS

load_dotenv()

log = logging.getLogger("voice-agent")
log.setLevel(logging.INFO)


# ───────────────────────────────────────────
# helpers
# ───────────────────────────────────────────

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


# ───────────────────────────────────────────
# histórico (opcional)
# ───────────────────────────────────────────

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
        log.warning("asyncpg não instalado — histórico desativado")
        return ""
    except Exception as exc:
        log.warning("Erro ao buscar histórico: %s", exc)
        return ""


# ───────────────────────────────────────────
# extrair número do cliente
# ───────────────────────────────────────────

def extrair_numero(room_name: str) -> str:
    for parte in room_name.split("_"):
        if parte.isdigit() and len(parte) >= 8:
            return parte
    return "desconhecido"


# ───────────────────────────────────────────
# agente com tools dentro da classe
# ───────────────────────────────────────────

class Assistente(Agent):
    def __init__(self, numero: str, historico: str):
        self._numero = numero

        prompt_base = env("AGENT_PROMPT", "Você é um atendente virtual simpático e objetivo.")
        partes = [
            prompt_base,
            "Seja breve e natural, como uma conversa por voz.",
            "Responda sempre em português brasileiro.",
        ]

        # informa ao LLM quais tools estão disponíveis
        tools_disponiveis = []
        if env_ok("N8N_WEBHOOK_URL"):
            tools_disponiveis.append("enviar_notificacao — envia notificação/recado ao sistema interno")
        if env_ok("AGENDA_WEBHOOK_URL"):
            tools_disponiveis.append("gerenciar_agenda — consulta, agenda ou cancela horários")
        if env_ok("LIGAR_URL"):
            tools_disponiveis.append("transferir_ligacao — transfere para atendente humano")

        if tools_disponiveis:
            partes.append("\nVocê tem acesso às seguintes ferramentas: " + ", ".join(tools_disponiveis) + ".")
            partes.append("Use-as quando fizer sentido para atender o cliente.")

        if historico:
            partes.append(
                f"\n--- Histórico recente ---\n{historico}\n"
                "Use isso apenas como contexto, não repita."
            )

        super().__init__(instructions="\n".join(partes))

    # ── N8N: enviar notificação/mensagem ──
    @function_tool(
        description=(
            "Envia uma notificação ou mensagem para o sistema interno (N8N). "
            "Use quando o cliente pedir para enviar um recado, notificação, "
            "registrar uma solicitação ou qualquer ação que precise ser processada externamente."
        ),
    )
    async def enviar_notificacao(
        self,
        ctx: RunContext,
        mensagem: str,
        assunto: str = "Solicitação do cliente",
    ) -> str:
        """
        Args:
            ctx: Contexto da execução.
            mensagem: Conteúdo da notificação ou solicitação.
            assunto: Resumo curto do motivo da notificação.
        """
        url = env("N8N_WEBHOOK_URL")
        if not url:
            return "Serviço de notificação não configurado no momento."
        log.info("Tool enviar_notificacao → %s", assunto)
        resultado = await post_webhook(url, {
            "numero": self._numero,
            "assunto": assunto,
            "mensagem": mensagem,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if "erro" in resultado:
            return f"Não consegui enviar a notificação: {resultado['erro']}"
        return "Notificação enviada com sucesso."

    # ── AGENDA: consultar / agendar horário ──
    @function_tool(
        description=(
            "Consulta horários disponíveis ou agenda um compromisso. "
            "Use quando o cliente quiser marcar, verificar ou cancelar um horário."
        ),
    )
    async def gerenciar_agenda(
        self,
        ctx: RunContext,
        acao: str,
        data_hora: str = "",
        observacao: str = "",
    ) -> str:
        """
        Args:
            ctx: Contexto da execução.
            acao: 'consultar', 'agendar' ou 'cancelar'.
            data_hora: Data/hora desejada (formato livre, ex: 'amanhã às 14h').
            observacao: Informação extra sobre o agendamento.
        """
        url = env("AGENDA_WEBHOOK_URL")
        if not url:
            return "Serviço de agenda não configurado no momento."
        log.info("Tool gerenciar_agenda → %s %s", acao, data_hora)
        resultado = await post_webhook(url, {
            "numero": self._numero,
            "acao": acao,
            "data_hora": data_hora,
            "observacao": observacao,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if "erro" in resultado:
            return f"Erro ao acessar a agenda: {resultado['erro']}"
        return json.dumps(resultado, ensure_ascii=False)

    # ── LIGAR: transferir / iniciar ligação ──
    @function_tool(
        description=(
            "Transfere a ligação para um atendente humano ou inicia uma chamada. "
            "Use quando o cliente pedir para falar com um humano, "
            "ou quando o assunto exigir atendimento especializado."
        ),
    )
    async def transferir_ligacao(
        self,
        ctx: RunContext,
        motivo: str,
        destino: str = "",
    ) -> str:
        """
        Args:
            ctx: Contexto da execução.
            motivo: Por que a ligação está sendo transferida.
            destino: Número ou setor de destino (opcional).
        """
        url = env("LIGAR_URL")
        if not url:
            return "Serviço de transferência não configurado no momento."
        log.info("Tool transferir_ligacao → %s", motivo)
        resultado = await post_webhook(url, {
            "numero": self._numero,
            "motivo": motivo,
            "destino": destino,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if "erro" in resultado:
            return f"Não foi possível transferir: {resultado['erro']}"
        return "Ligação sendo transferida. Aguarde um momento."


# ───────────────────────────────────────────
# entrypoint
# ───────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    numero = extrair_numero(ctx.room.name or "")
    log.info("Sala: %s | Cliente: %s", ctx.room.name, numero)

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


# ───────────────────────────────────────────
# run
# ───────────────────────────────────────────

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
