from langchain.agents import create_agent
from src.config.fundamental_models import llm_qwen3
from src.agents.planner.prompt import SYSTEM_PROMPT

planner_agent = create_agent(
    model=llm_qwen3,
    system_prompt=SYSTEM_PROMPT,
)
