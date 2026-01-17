from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from src.agents.communicator.prompt import COMMUNICATOR_PROMPT
from src.config.fundamental_models import llm_qwen3

communicator_agent = create_agent(
    model=llm_qwen3,
    system_prompt=COMMUNICATOR_PROMPT,
    store=InMemoryStore(),
    #response_format=ToolStrategy(CollectionResult),
)
