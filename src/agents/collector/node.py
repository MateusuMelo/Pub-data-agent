# src/agents/collector/node.py
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage

from agents.collector.tools import ibge_agregados_request
from src.agents.collector.agent import collector_agent
from src.agents.collector.schema import CollectionResult
from src.agents.state import AgentState

logger = logging.getLogger(__name__)


def collector_node(state: AgentState) -> AgentState:
    """
    Collector node responsible for executing the collector agent
    and storing a CollectionResult in state["data"].
    """
    logger.info("🚜 Collector node started")

    execution_plan = state.get("execution_plan", [])
    current_step = state.get("current_step", 0)

    # 🔎 localizar passo do collector
    collector_step = next(
        (step for step in execution_plan if step.get("agent") == "collector"),
        None
    )

    if not collector_step:
        raise ValueError("Collector step not found in execution plan")

    task = collector_step.get("task", "")
    parameters = collector_step.get("parameters", {})

    logger.info("📥 Collector task: %s", task)
    logger.debug("Collector parameters: %s", parameters)

    # 🧠 última pergunta do usuário
    user_question = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            user_question = msg.content
            break

    # 📦 contexto enviado ao agent
    payload = {
        "task": task,
        "parameters": parameters,
        "user_question": user_question,
    }

    try:
        # GET ASSUNTO
        assunto_agent_input = {
            "messages": [
                HumanMessage(
                    content=f"Voce deve procurar qual assunto mais relacionado a: '{payload['parameters']['concept']}' baseado em seus conhecimentos. Sua resposta deve ser somente :'id'"),
            ]
        }
        get_assunto_response = collector_agent.invoke(assunto_agent_input)
        assunto_id = _parse_assunto_collector_result(get_assunto_response)

        # FIX: Pass as dictionary instead of integer
        agregados_input = {"assunto_id": assunto_id}  # Adjust key based on what the tool expects
        agregados = ibge_agregados_request.invoke(agregados_input)

        # GET Agregado id
        agregado_agent_input = {
            "messages": [
                HumanMessage(
                    content=f"Dentre estes agregados {agregados['temas_encontrados'][0]['agregados']} identifique o unico que mais se relaciona com o objetivo :{task}. Sua resposta deve ser somente :'id'"),
            ]
        }
        get_agregado_response = collector_agent.invoke(agregado_agent_input)
        agregados_id = _parse_assunto_collector_result(get_agregado_response)
        # Create CollectionResult with the collected data
        collection_result = CollectionResult(
            success=True,
            collected_data=[agregados],  # Or format as needed
            failed_variables=[],
            errors=[],
            metadata={"task": task, "assunto_id": assunto_id},
            source_used={"name": "IBGE", "description": f"Agregados for assunto {assunto_id}"},
            filters_applied=parameters.get("filters", {}),
            collection_time=datetime.utcnow().isoformat() + "Z",
        )

        logger.info(
            "✅ Collector finished | success=%s | collected=%d | failed=%d",
            collection_result.success,
            len(collection_result.collected_data),
            len(collection_result.failed_variables),
        )

        return {
            "messages": state.get("messages", []),
            "execution_plan": execution_plan,
            "current_step": current_step + 1,
            "data": collection_result.model_dump(),
            "analysis": state.get("analysis"),
            "answer": state.get("answer", ""),
        }

    except Exception as e:
        logger.exception("❌ Collector node failed")

        failed = CollectionResult(
            success=False,
            collected_data=[],
            failed_variables=parameters.get("variables", []),
            errors=[str(e)],
            metadata={"task": task},
            source_used={"name": "collector", "description": "execution error"},
            filters_applied=parameters.get("filters", {}),
            collection_time=datetime.utcnow().isoformat() + "Z",
        )

        return {
            "messages": state.get("messages", []),
            "execution_plan": execution_plan,
            "current_step": current_step + 1,
            "data": failed.model_dump(),
            "analysis": state.get("analysis"),
            "answer": state.get("answer", ""),
        }


# -------------------------------------------------------------------------
# 🔧 Helpers
# -------------------------------------------------------------------------
import json
from typing import Any, Optional


def _parse_assunto_collector_result(result: Any) -> Optional[int]:
    """
    Parse collector result and extract the 'id' returned by the final AIMessage.

    Rules:
    - Only parse the final AIMessage content.
    - Content must be valid JSON.
    - Must contain the key 'id'.
    - Do NOT infer, fallback, or inspect tool messages.
    """

    if not result or not isinstance(result, dict):
        return None

    messages = result.get("messages")
    if not isinstance(messages, list):
        return None

    # Iterate in reverse to find the last AIMessage with content
    for message in reversed(messages):
        if message.__class__.__name__ != "AIMessage":
            continue

        content = getattr(message, "content", None)
        if not content or not isinstance(content, str):
            continue

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict) and "id" in payload:
            return payload["id"]

    return None


def _parse_collector_result(result: Any) -> CollectionResult:
    """
    Extract CollectionResult from LangChain agent output.

    Supported formats:
      1) AIMessage.parsed (ToolStrategy happy path)
      2) JSON string in content
      3) <CollectionResult ... /> XML-like fallback
    """

    # Caso raro: já veio validado
    if isinstance(result, CollectionResult):
        return result

    if not isinstance(result, dict) or "messages" not in result:
        raise ValueError("Invalid collector agent response format")

    messages = result["messages"]

    # Pegamos a ÚLTIMA mensagem do modelo
    last_msg = messages[-1]

    if not isinstance(last_msg, AIMessage):
        raise ValueError("Last message is not AIMessage")

    # ─────────────────────────────
    # 1️⃣ Caminho feliz: ToolStrategy
    # ─────────────────────────────
    if hasattr(last_msg, "parsed") and isinstance(last_msg.parsed, CollectionResult):
        return last_msg.parsed

    content = (last_msg.content or "").strip()
    if not content:
        raise ValueError("AIMessage content is empty")

    # ─────────────────────────────
    # 2️⃣ JSON puro
    # ─────────────────────────────
    try:
        data = json.loads(content)
        return CollectionResult.model_validate(data)
    except json.JSONDecodeError:
        pass

    # ─────────────────────────────
    # 3️⃣ XML-like <CollectionResult ... />
    # ─────────────────────────────
    if content.startswith("<collectionResult"):
        return _parse_collection_result_from_xml_like(content)

    raise ValueError("Unable to extract CollectionResult from collector response")


def _parse_collection_result_from_xml_like(text: str) -> CollectionResult:
    """
    Parse <CollectionResult ... /> output produced by the LLM.
    """

    def _extract_attr(name: str):
        match = re.search(rf'{name}="([^"]+)"', text)
        return match.group(1) if match else None

    def _extract_json_attr(name: str):
        raw = _extract_attr(name)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return None

    return CollectionResult(
        success=_extract_attr("success") == "true",
        source=_extract_attr("source"),

        identified_parameters=_extract_json_attr("identified_parameters"),
        params=_extract_json_attr("params"),
        raw_data=_extract_json_attr("raw_data"),
        metadata=_extract_json_attr("metadata"),

        errors=None,
    )
