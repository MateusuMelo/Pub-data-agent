from langgraph.graph import END
from src.agents.state import AgentState


def route_by_plan(state: AgentState):
    step_index = state["current_step"]

    if step_index >= len(state["execution_plan"]):
        return END

    return state["execution_plan"][step_index]["agent"]
