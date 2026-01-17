import logging
from langchain_core.messages import HumanMessage
from src.agents.state import AgentState
from src.workflow.graph import construct_graph
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

initial_state: AgentState = {
    "messages": [
        HumanMessage(
            content="Quantidade populacional por região"
        )
    ],
    "execution_plan": [],
    "current_step": 0,
    "data": None,
    "analysis": None,
    "answer": "",
}


graph = construct_graph()

updated_state = graph.invoke(initial_state)

print("\n--- FINAL STATE (after planner) ---")
print(updated_state)
