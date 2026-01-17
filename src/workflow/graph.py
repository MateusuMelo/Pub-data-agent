from langgraph.constants import END
from langgraph.graph import StateGraph

from src.agents.communicator.node import communicator_node
from src.agents.collector.node import collector_node
from src.agents.planner.node import planner_node
from src.agents.state import AgentState

def construct_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node('collector', collector_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "collector")
    g.add_edge("collector", END)

    return g.compile()

