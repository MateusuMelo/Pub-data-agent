from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    execution_plan: List[Dict[str, Any]]
    current_step: int
    data: Any
    analysis: Any
    answer: str
