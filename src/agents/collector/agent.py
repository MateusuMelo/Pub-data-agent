from functools import lru_cache

from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

from src.agents.collector.tools import (
    ibge_documentation_search,
    ibge_assunto_id_search,
    ibge_classificacao_id_search,
    ibge_agregados_request,
    ibge_nivel_geografico_id_search,
)
from src.agents.collector.prompt import COLLECTOR_PROMPT
from src.config.fundamental_models import llm_qwen3


@lru_cache(maxsize=1)
def get_collector_agent():
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
            ibge_agregados_request,
        ],
        store=InMemoryStore(),
    )


# Backwards-compatibility export while we migrate callers
collector_agent = get_collector_agent()
