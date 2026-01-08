from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from src.agents.collector.tools import (
    ibge_documentation_search,
    ibge_assunto_id_search,
    ibge_classificacao_id_search,
    ibge_agregados_request
)

from src.agents.collector.prompt import COLLECTOR_PROMPT
from src.agents.collector.schema import CollectionResult
from src.config.fundamental_models import llm_qwen3

collector_agent = create_agent(
    model=llm_qwen3,
    system_prompt=COLLECTOR_PROMPT,
    tools=[
        # 1️⃣ Entendimento
        ibge_documentation_search,
        ibge_assunto_id_search,
        ibge_classificacao_id_search,

        ibge_agregados_request,
    ],
    store=InMemoryStore(),
    response_format=ToolStrategy(CollectionResult),
)
