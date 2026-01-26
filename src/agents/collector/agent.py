from functools import lru_cache

from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

from agents.collector.schema import AgregadoID
from src.agents.collector.tools import (
    ibge_documentation_search,
    ibge_assunto_id_search,
    ibge_classificacao_id_search,
    ibge_agregados_request,
    ibge_nivel_geografico_id_search,
)
from src.agents.collector.prompt import COLLECTOR_PROMPT
from src.config.fundamental_models import llm_qwen3, llm_qwen3_m


@lru_cache(maxsize=4)
def get_collector_agent_singleton():
    """
    Factory/Singleton for the collector agent to avoid multiple global instances
    and centralize construction following the Factory pattern.
    """
    return create_agent(
        model=llm_qwen3,
        system_prompt=COLLECTOR_PROMPT,
        tools=[
            ibge_documentation_search,
            ibge_assunto_id_search,
            ibge_classificacao_id_search,
            ibge_agregados_request,
            ibge_nivel_geografico_id_search
        ],
        store=InMemoryStore(),
    )


def get_collector_agent_multi():
    return llm_qwen3_m.with_structured_output(
        AgregadoID,
        method="json_schema",
        strict=True
    )

# Backwards-compatibility export while we migrate callers
collector_agent_singleton = get_collector_agent_singleton()
collector_agent_multi = get_collector_agent_multi()
