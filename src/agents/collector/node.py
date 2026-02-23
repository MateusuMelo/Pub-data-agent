# src/agents/collector/node.py
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from src.agents.collector.tools import ibge_agregados_request, ibge_agregado_metadados_request, \
    ibge_agregado_dados_request, \
    ibge_nivel_geografico_id_search, ibge_agregado_id_search
from src.agents.collector.agent import collector_agent_s, collector_agent_m
from src.agents.collector.schema import CollectionResult, CollectorCompleteResult, AgregadoID
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
    user_question = extract_last_user_question(state.get("messages", []))

    try:

        # 3. Selecionar agregado específico
        agregados_id = select_agregado_id(user_question)
        logger.info(f"🎯 Agregado selecionado: {agregados_id}")

        # 4. Obter metadados do agregado
        agregados_metadados = get_agregado_metadata(agregados_id)
        logger.info("📄 Metadados do agregado obtidos")

        # 5. Selecionar período
        periodo_id = select_periodo_id(agregados_metadados, task)
        logger.info(f"📅 Período selecionado: {periodo_id}")

        # 6. Selecionar território
        territorio_id = select_territorio_id(agregados_metadados, task)
        logger.info(f"🗺️ Território selecionado: {territorio_id}")

        # 7. Selecionar variável
        variavel_id = select_variavel_id(agregados_metadados, task)
        logger.info(f"📈 Variável selecionada: {variavel_id}")

        # 8. Obter classificação
        classificacao_id = get_classificacao_id(agregados_metadados)
        logger.info(f"🏷️ Classificação selecionada: {classificacao_id}")

        # 9. Obter dados finais
        results = get_ibge_data(
            agregados_id=agregados_id,
            periodo_id=periodo_id,
            variavel_id=variavel_id,
            territorio_id=territorio_id,
            classificacao_id=classificacao_id
        )
        logger.info(f"✅ Dados coletados com sucesso")

        df = ibge_results_to_dataframe(
            results=results if isinstance(results, list) else [results],
            assunto_nome=f"{collector_step['parameters']['concept']}_{collector_step['parameters']['territory']}".replace(
                " ", "_")
        )

        # Create CollectorCompleteResult com todos os dados para o próximo agente
        collection_result = CollectorCompleteResult(
            success=True,
            collected_data=[results] if results else [],
            failed_variables=[],
            errors=[],
            metadata={
                "task": task,
             #   "assunto_id": assunto_id,
                "agregado_id": agregados_id,
                "periodo_id": periodo_id,
                "territorio_id": territorio_id,
                "variavel_id": variavel_id,
                "classificacao_id": classificacao_id
            },
            filters_applied=parameters.get("filters", {}),
            task=task,
            parameters=parameters,
            #assunto_id=assunto_id,
            agregado_id=agregados_id,
            periodo_id=periodo_id,
            territorio_id=territorio_id,
            variavel_id=variavel_id,
            classificacao_id=classificacao_id,
            raw_dados=results if isinstance(results, list) else [results]
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
            "data": collection_result.model_dump(),  # Agora é um dicionário completo
            "analysis": state.get("analysis"),
            "answer": state.get("answer", ""),
        }

    except Exception as e:
        logger.exception("❌ Collector node failed")

        # Criar resultado de erro também como CollectorCompleteResult
        failed = CollectorCompleteResult(
            success=False,
            collected_data=[],
            failed_variables=parameters.get("variables", []),
            errors=[str(e)],
            metadata={"task": task},
            source_used={"name": "collector", "description": "execution error"},
            filters_applied=parameters.get("filters", {}),
            task=task,
            parameters=parameters,
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
# 🔧 FUNÇÕES AUXILIARES PRINCIPAIS
# -------------------------------------------------------------------------


def ibge_results_to_dataframe(
        results: List[Dict[str, Any]],
        assunto_nome: str,
        output_dir: str = "data/ibge"
) -> pd.DataFrame:
    """
    Transforma a saída bruta do IBGE/SIDRA em um DataFrame tabular
    e salva o arquivo com nome baseado no assunto e data da coleta.
    """

    rows = []

    for var in results:
        var_id = var.get("id")
        var_nome = var.get("variavel")
        unidade = var.get("unidade")

        for resultado in var.get("resultados", []):
            classificacoes = resultado.get("classificacoes", [])

            for classificacao in classificacoes:
                class_id = classificacao.get("id")
                class_nome = classificacao.get("nome")

                categoria = classificacao.get("categoria", {})
                categoria_id, categoria_nome = None, None
                if isinstance(categoria, dict) and categoria:
                    categoria_id, categoria_nome = next(iter(categoria.items()))

                for serie_item in resultado.get("series", []):
                    local = serie_item.get("localidade", {})
                    local_id = local.get("id")
                    local_nome = local.get("nome")

                    nivel = local.get("nivel")
                    nivel_id = nivel.get("id") if nivel else None
                    nivel_nome = nivel.get("nome") if nivel else None

                    serie = serie_item.get("serie", {})

                    for periodo, valor in serie.items():
                        rows.append({
                            "variavel_id": var_id,
                            "variavel": var_nome,
                            "unidade": unidade,
                            "classificacao_id": class_id,
                            "classificacao": class_nome,
                            "categoria_id": categoria_id,
                            "categoria": categoria_nome,
                            "localidade_id": local_id,
                            "localidade": local_nome,
                            "nivel_id": nivel_id,
                            "nivel": nivel_nome,
                            "periodo": periodo,
                            "valor": None if valor in ("...", None) else float(valor)
                        })

    df = pd.DataFrame(rows)

    # ─────────────────────────────
    # 📁 Preparar nome do arquivo
    # ─────────────────────────────

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    assunto_slug = re.sub(r"[^a-zA-Z0-9]+", "_", assunto_nome.lower()).strip("_")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    filename = f"{assunto_slug}_{timestamp}.csv"
    filepath = Path(output_dir) / filename

    df.to_csv(filepath, index=False)

    return df


def extract_last_user_question(messages: list) -> str:
    """Extrai a última pergunta do usuário."""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return ""


def get_assunto_id(concept: str) -> int:
    """Obtém o ID do assunto relacionado ao conceito."""
    assunto_agent_input = {
        "messages": [
            HumanMessage(
                content=(
                    f"Find the subject as the concept: '{concept}'.\n"
                    "You MUST call the ibge_agregado_id_search tool.\n"
                    "Calling this tool more than once is FORBIDDEN.\n"
                    "You MUST select only a subject that matches the SAME meaning, not a related or approximate one.\n"
                    "After calling the tool, you MUST stop.\n"
                    "Your FINAL output MUST be a valid JSON object.\n"
                    "Do NOT include explanations, markdown, comments, or extra text.\n"
                    "Return EXACTLY this format:\n"
                    "{\"id\": \"<ID_AGREGADO>\"}\n"
                    "This is a FINAL answer. Do not perform any additional reasoning or actions."
                )
            ),
        ]
    }

    response = collector_agent_s.invoke(assunto_agent_input)
    assunto_id = _parse_assunto_collector_result(response)

    if assunto_id is None:
        raise ValueError("Não foi possível encontrar o assunto ID")

    return assunto_id


def get_agregados_from_assunto(assunto_id: int) -> dict:
    """Obtém agregados relacionados a um assunto."""
    agregados_input = {"assunto_id": assunto_id}
    return ibge_agregados_request.invoke(agregados_input)


def select_agregado_id(task: str) -> int:
    """Seleciona o agregado mais relevante para a tarefa."""

    agregados_list = ibge_agregado_id_search.invoke({"query": task})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.\n"
                "Your job is to select the correct IBGE agregado id.\n"
                "You MUST follow all rules strictly."
            ),
            (
                "system",
                "CONTEXT:\n"
                "Below is a list of IBGE agregados already retrieved.\n"
                "You MUST choose ONLY from this list.\n\n"
                "Agregados list (JSON):\n"
                "{agregados_list}"
            ),
            (
                "system",
                "RULES:\n"
                "- You MUST select the agregado with the EXACT SAME meaning.\n"
                "- Do NOT select related, broader, narrower, or approximate concepts.\n"
                "- You MUST NOT call any tool.\n"
                "- You MUST NOT invent or modify ids.\n"
                "- You MUST return an id that exists in the provided list.\n"
                "- Your FINAL output MUST be a valid JSON object.\n"
                "- Do NOT include explanations, markdown, comments, or extra text.\n\n"
                "Return EXACTLY this format:\n"
                "{{\"id\": \"<id>\"}}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Task concept to match:\n\"{task}\""
            ),
        ]
    )

    messages = prompt.format_messages(
        task=task,
        agregados_list=agregados_list,
        chat_history=[],
    )

    response: AgregadoID = collector_agent_m.invoke(messages)


    agregado_id = _parse_assunto_collector_result(response)
    agregado_id = int(agregado_id) if isinstance(agregado_id, str) else agregado_id

    if agregado_id is None:
        raise ValueError("Não foi possível selecionar o agregado")

    return agregado_id

def get_agregado_metadata(agregado_id: int) -> dict:
    """Obtém metadados completos de um agregado."""
    agregados_input = {"agregado_id": agregado_id}
    return ibge_agregado_metadados_request.invoke(agregados_input)


def select_periodo_id(agregados_metadados: dict, task: str) -> int:
    """Seleciona o período mais relevante para a tarefa."""
    periodos_list = agregados_metadados.get('periodos_disponiveis', {}).get('periodos', [])

    if not periodos_list:
        raise ValueError("Nenhum período disponível para este agregado")

    select_periodo_input = {
        "messages": [
            HumanMessage(
                content=(
                    f"Among the following periods {periodos_list}, select the SINGLE one that is most related to the objective: {task}.\n"
                    "If the objective does not contain any period-related instructions, select the LAST period.\n"
                    "You are FORBIDDEN from calling any tool.\n"
                    "Your output MUST be a valid JSON object.\n"
                    "Do NOT include explanations, markdown, comments, or extra text.\n"
                    "Return EXACTLY this format:\n"
                    "{\"id\": \"<PERIODO_ID>\"}\n"
                    "This is a FINAL answer. Do not perform any additional actions."
                )
            ),
        ]
    }

    response = collector_agent_s.invoke(select_periodo_input)
    periodo_id = _parse_assunto_collector_result(response)

    # Garantir que é inteiro
    if isinstance(periodo_id, str) and periodo_id.isdigit():
        periodo_id = int(periodo_id)
    elif periodo_id is None:
        # Selecionar o último período como fallback
        periodo_id = periodos_list[-1]
        if isinstance(periodo_id, str) and periodo_id.isdigit():
            periodo_id = int(periodo_id)

    if not isinstance(periodo_id, int):
        raise ValueError(f"Período inválido: {periodo_id}")

    return periodo_id


def select_territorio_id(agregados_metadados: dict, task: str) -> str:
    """Seleciona o território mais relevante para a tarefa."""

    nivel_territorial = agregados_metadados.get("metadados", {}).get("nivelTerritorial", {})

    if not nivel_territorial:
        raise ValueError("Nenhum nível territorial disponível para este agregado")

    # 1. Extrair todos os códigos territoriais
    all_territorial_codes = []

    if isinstance(nivel_territorial, dict):
        for value in nivel_territorial.values():
            if isinstance(value, list):
                all_territorial_codes.extend(value)
            elif value:
                all_territorial_codes.append(value)
    elif isinstance(nivel_territorial, list):
        all_territorial_codes = nivel_territorial

    all_territorial_codes = [
        str(code).strip()
        for code in all_territorial_codes
        if code and str(code).strip()
    ]
    all_territorial_codes = list(dict.fromkeys(all_territorial_codes))

    if not all_territorial_codes:
        raise ValueError("Nenhum código territorial válido encontrado")

    territorios_detalhados = []

    for code in all_territorial_codes:
        tool_input = {"query": code}
        tool_result = ibge_nivel_geografico_id_search.invoke(tool_input)

        territorios_detalhados.append(tool_result)

    if not territorios_detalhados:
        raise ValueError("Não foi possível resolver nenhum código territorial via ferramenta")

    select_territorio_input = {
        "messages": [
            HumanMessage(
                content=(
                    "RESOLVED CONTEXT (TERRITORIAL LEVELS):\n\n"
                    "The following territorial codes have already been resolved via IBGE:\n\n"
                    f"{territorios_detalhados}\n\n"

                    "TASK:\n"
                    f"{task}\n\n"

                    "INSTRUCTIONS:\n"
                    "- Analyze the objective of the task\n"
                    "- Choose the MOST SPECIFIC and MOST APPROPRIATE territorial level\n"
                    "- Return ONLY ONE code\n"
                    "- DO NOT call any tools\n"
                    "- DO NOT perform searches or external queries\n"
                    "- Use ONLY the information provided in the resolved context above\n\n"

                    "EXACT OUTPUT FORMAT (pure JSON):\n"
                    r'{"id": "N<number>"}'
                )
            )
        ]
    }

    # 4. Agente só decide
    response = collector_agent_s.invoke(select_territorio_input)
    territorio_id = _parse_territorio_collector_result(response)

    # 5. Fallback
    if territorio_id is None:
        territorio_id = "N1[all]"  # Brasil como padrão

    return territorio_id


def select_variavel_id(agregados_metadados: dict, task: str) -> int:
    """Seleciona a variável mais relevante para a tarefa."""
    variaveis = agregados_metadados.get('metadados', {}).get('variaveis', [])

    if not variaveis:
        raise ValueError("Nenhuma variável disponível para este agregado")

    select_variavel_input = {
        "messages": [
            HumanMessage(
                content=f"Dentre estas variaveis {variaveis} identifique a unica que mais se relaciona com o objetivo: {task}.\n"
                        "Voce não deve chamar nenhuma ferramenta.\n"
                        "Sua resposta deve ser somente: {{\"id\":{{id_variavel}}}}"
            ),
        ]
    }
    response = collector_agent_s.invoke(select_variavel_input)
    variavel_id = _parse_assunto_collector_result(response)

    if variavel_id is None:
        raise ValueError("Não foi possível selecionar a variável")

    return variavel_id


def get_classificacao_id(agregados_metadados: dict) -> Optional[int]:
    """Obtém a primeira classificação disponível."""
    classificacoes = agregados_metadados.get('metadados', {}).get('classificacoes', [])

    if classificacoes and len(classificacoes) > 0:
        return classificacoes[0].get('id')

    return None


def get_ibge_data(
        agregados_id: int,
        periodo_id: int,
        variavel_id: int,
        territorio_id: str,
        classificacao_id: Optional[int]
) -> Any:
    """Obtém os dados finais da API IBGE."""

    # Garantir que territorio_id não seja None
    if territorio_id is None:
        territorio_id = "N1[all]"

    params = {
        "agregado": agregados_id,
        "periodo": periodo_id,
        "variavel": variavel_id,
        "territorio": territorio_id,
    }

    # Adicionar classificação apenas se existir
    if classificacao_id is not None:
        params["classificacao"] = classificacao_id

    return ibge_agregado_dados_request.invoke(params)


# -------------------------------------------------------------------------
# 🔧 HELPER FUNCTIONS PARA PARSING
# -------------------------------------------------------------------------

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


def parse_nivel_geografico_response(raw: str) -> dict[str, str]:
    """
    Parseia a resposta textual da ferramenta ibge_nivel_geografico_id_search
    para um dicionário {codigo: descricao}.
    """
    result = {}

    # Remove quebras estranhas e normaliza espaços
    normalized = re.sub(r"\s+", " ", raw.strip())

    # Regex: captura código e descrição
    pattern = re.compile(
        r"NIVEL_GEOGRAFICO:\s*(N\d+)\s*\|\s*([^N]+?)(?=NIVEL_GEOGRAFICO:|$)"
    )

    for code, desc in pattern.findall(normalized):
        result[code] = desc.strip()

    return result


def _parse_territorio_collector_result(result: Any) -> Optional[str]:
    """
    Parse collector result and extract territorio ID (string) from the final AIMessage.
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

        clean_content = content.strip()
        if not clean_content:
            continue

        try:
            payload = json.loads(clean_content)
        except json.JSONDecodeError:
            # Try to find JSON pattern in the content
            json_match = re.search(r'\{[^{}]*"id"[^{}]*:[^{}]*[^{}]*\}', clean_content)
            if json_match:
                try:
                    payload = json.loads(json_match.group())
                except:
                    continue
            else:
                continue

        if isinstance(payload, dict) and "id" in payload:
            id_value = payload["id"]
            if isinstance(id_value, str):
                return id_value
            elif isinstance(id_value, (int, float)):
                return str(id_value)

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
