"""
Agent initialization and setup.
"""
from services.llm_service import LLMService
from agents.coordinator.coordinator import CoordinatorAgent
from agents import (
    MoleculePlotAgent, NMRPlotAgent, TextResponseAgent,
    ToolAgent, OrchestrationAgent, AgentType
)

# Initialize LLM service
llm_service = LLMService()

if not llm_service:
    raise RuntimeError("No LLM service available. Please check your API keys in config/settings.py")

# Initialize coordinator first
agent_coordinator = CoordinatorAgent(llm_service)

# Initialize all other agents
molecule_plot_agent = MoleculePlotAgent(llm_service)
nmr_plot_agent = NMRPlotAgent(llm_service)
text_response_agent = TextResponseAgent(llm_service)
tool_agent = ToolAgent(llm_service)

# Initialize orchestration agent with its coordinator
orchestration_agent = OrchestrationAgent(llm_service, coordinator=agent_coordinator)

# Register all agents with coordinator
agent_coordinator.add_agent(AgentType.MOLECULE_PLOT, molecule_plot_agent)
agent_coordinator.add_agent(AgentType.NMR_PLOT, nmr_plot_agent)
agent_coordinator.add_agent(AgentType.TEXT_RESPONSE, text_response_agent)
agent_coordinator.add_agent(AgentType.TOOL_USE, tool_agent)
agent_coordinator.set_orchestration_agent(orchestration_agent)
