import json
import logging
from src.agents.planner.schema import ExecutionPlan
from src.agents.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.planner.agent import planner_agent

logger = logging.getLogger(__name__)


def planner_node(state: AgentState):
    logger.info("Planner node started")

    user_question = state["messages"][-1].content
    logger.info("User question: %s", user_question)

    # O agente já retorna o formato estruturado
    agent_input = {
        "messages": [HumanMessage(content=user_question)]
    }

    try:
        result = planner_agent.invoke(agent_input)

        # O resultado já deve ser um objeto ExecutionPlan
        # Dependendo da implementação do create_agent, pode estar em diferentes locais
        if hasattr(result, 'model_dump'):
            # Se retornar diretamente o modelo Pydantic
            plan_data = result.model_dump()
        elif isinstance(result, dict) and 'output' in result:
            # Se estiver em result['output']
            plan_data = result['output']
        elif isinstance(result, dict) and 'messages' in result:
            # Se ainda estiver nas mensagens (fallback)
            ai_message = result['messages'][-1]
            if hasattr(ai_message, 'parsed'):
                # Se o LangChain parseou automaticamente
                plan_data = ai_message.parsed
            else:
                # Parse manual como fallback
                plan_data = json.loads(ai_message.content)
        else:
            plan_data = result

        # Se for um dict, converter para ExecutionPlan para validação
        if isinstance(plan_data, dict):
            execution_plan = ExecutionPlan.model_validate(plan_data)
        else:
            execution_plan = plan_data

        logger.info(
            "Planner generated %d validated steps",
            len(execution_plan.execution_plan)
        )

        # Log dos steps
        for i, step in enumerate(execution_plan.execution_plan):
            logger.debug("Step %d: %s - %s", i, step.agent, step.task)
            if step.parameters:
                logger.debug("  Parameters: %s", step.parameters)

        return {
            "execution_plan": [step.model_dump() for step in execution_plan.execution_plan],
            "current_step": 0
        }

    except Exception as e:
        logger.error("Planner failed: %s", str(e))
        raise ValueError(f"Planner error: {str(e)}")