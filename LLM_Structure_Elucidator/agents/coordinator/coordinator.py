"""
Main agent coordinator/orchestrator for managing agent interactions.
"""
from typing import Dict, Any, List, Optional, Tuple
from ..base import BaseAgent
from services.llm_service import LLMService
import json
import traceback
from enum import Enum, auto

class AgentType(Enum):
    """Available agent types in the system."""
    MOLECULE_PLOT = auto()
    NMR_PLOT = auto()
    TEXT_RESPONSE = auto()
    TOOL_USE = auto()  # Added tool agent type
    ORCHESTRATION = auto()  # Added orchestration agent type
    ANALYSIS = auto()  # Added analysis agent type

class CoordinatorAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.agents: Dict[AgentType, BaseAgent] = {}
        
        # Initialize tool agent
        from ..specialized.tool_agent import ToolAgent
        self.tool_agent = ToolAgent(llm_service)
        self.tools = self.tool_agent.tools
        
        # Initialize analysis agent
        from ..specialized.analysis_agent import AnalysisAgent
        self.analysis_agent = AnalysisAgent(llm_service)
        
        # Register agents
        self.add_agent(AgentType.TOOL_USE, self.tool_agent)
        self.add_agent(AgentType.ANALYSIS, self.analysis_agent)
        
        # Define agent descriptions for LLM
        self._init_agent_descriptions()
        
    def set_orchestration_agent(self, orchestration_agent):
        """Set the orchestration agent and register it."""
        self.orchestration_agent = orchestration_agent
        self.add_agent(AgentType.ORCHESTRATION, orchestration_agent)
        
    def _init_agent_descriptions(self):
        """Initialize agent descriptions with string keys for JSON serialization."""
        self.agent_descriptions = {
            "MOLECULE_PLOT": {
                "name": "Molecule Plot",
                "description": "Specialized agent for handling molecular structure visualization requests",
                "capabilities": [
                    "2D/3D molecular structure visualization",
                    "Chemical structure drawing",
                    "Molecule rendering",
                ],
                "keywords": [
                    "molecule", "structure", "2D", "3D", 
                    "draw", "show", "visualize",
                    "can you show", "can you display",
                    "i want to see molecule"
                ]
            },
            "NMR_PLOT": {
                "name": "NMR Plot",
                "description": "Specialized agent for handling NMR spectroscopic data visualization and analysis",
                "capabilities": [
                    "1D/2D NMR spectrum visualization",
                    "HSQC plot generation",
                    "COSY correlation analysis",
                    "Peak analysis and integration",
                    "Chemical shift visualization"
                ],
                "keywords": [
                    "NMR", "spectrum", "HSQC", "COSY", 
                    "chemical shift", "proton", "carbon", 
                    "correlation", "peak", "integration",
                    "1h", "13c"
                ]
            },
            "ORCHESTRATION": {
                "name": "Workflow Orchestrator",
                "description": "Specialized agent for managing and executing structure elucidation workflows",
                "capabilities": [
                    "Structure elucidation workflow execution",
                    "Starting material workflow processing",
                    "Target structure workflow processing",
                    "Spectral-only workflow processing",
                    "Multi-step workflow coordination",
                    "Result validation and confidence scoring",
                    "Dynamic error threshold calculation",
                    "Forward and retrosynthesis prediction"
                ],
                "keywords": [
                    "analyze structure", "elucidate", "workflow",
                    "process spectra", "starting material",
                    "target structure", "analyze spectra",
                    "run analysis", "process workflow",
                    "structure analysis", "confidence score",
                    "run structure elucidation", "elucidation workflow",
                    "structure determination"
                ]
            },
            "TOOL_USE": {
                "name": "Tool Agent",
                "description": "Specialized agent for managing and coordinating various chemical prediction tools and simulations",
                "capabilities": [
                    "Retrosynthesis prediction and analysis",
                    "Forward synthesis prediction",
                    "Starting material identification",
                    "Reaction pathway prediction",
                    "Tool selection and coordination",
                    "NMR spectrum simulation",
                    "Analysis tool management",
                    "Simulation execution",
                    "Tool-specific task routing",
                    "Threshold calculation for NMR spectra",
                    "Peak matching and comparison",
                    "MMST structure improvement cycles",
                    "Multi-modal spectral analysis",
                    "Molecular analogue generation and scoring"
                ],
                "keywords": [
                    "starting material", "retrosynthesis", "predict starting materials",
                    "calculate starting material", "find starting materials",
                    "forward synthesis", "predict products", "reaction prediction",
                    "reaction pathway", "synthetic route", "synthesis steps",
                    "simulate", "prediction", "tool", "analysis",
                    "run", "execute", "predict", "generate",
                    "use tool", "run simulation", "analyze",
                    "nmr spectrum", "hsqc", "cosy",
                    "chemical shift", "proton", "carbon",
                    "threshold", "calculate threshold", "error threshold",
                    "peak matching", "compare peaks", "match spectra", "match peaks",
                    "run peak matching", "run nmr peak matching",
                    "improve structure", "structure improvement", "mmst",
                    "improvement cycle", "improve", "optimize structure",
                    "structure optimization", "molecular optimization"
                ]
            },
            "TEXT_RESPONSE": {
                "name": "Text Response",
                "description": "Default conversation handler for all queries that don't specifically require specialized agents",
                "capabilities": [
                    "Natural language understanding and conversation",
                    "General question answering on any topic",
                    "Chemistry-related discussions not requiring visualization",
                    "Contextual responses and clarifications",
                    "Information retrieval and explanation",
                    "Casual conversation and greetings",
                    "Fallback handling when other agents are not confident"
                ],
                "keywords": [
                    "what", "how", "why", "explain", 
                    "tell me", "describe", "help",
                    "question", "answer", "information", "hello",
                    "hi", "hey", "thanks", "thank you",
                    "can you", "could you", "please",
                    "discuss", "talk about", "understand"
                ]
            },
            "ANALYSIS": {
                "name": "Analysis Agent",
                "description": "Specialized agent for comprehensive analysis of molecular data and generation of interpretable results",
                "capabilities": [
                    "NMR data analysis and interpretation",
                    "Structure-spectrum correlation analysis",
                    "Visual molecular comparison analysis",
                    "LLM-based data interpretation",
                    "Data aggregation and reporting",
                    "Analysis coordination",
                    "Result explanation and summarization",
                    "Confidence scoring and validation",
                    "Discrepancy identification and explanation",
                    "Structure validation through spectral data"
                ],
                "keywords": [
                    "analyze data", "interpret results", "explain analysis",
                    "compare structures", "validate structure",
                    "analyze spectra", "interpret spectra",
                    "explain differences", "summarize results",
                    "explain discrepancies", "correlation analysis",
                    "confidence analysis", "structure validation",
                    "data interpretation", "comprehensive analysis",
                    "detailed analysis", "explain findings"
                ]
            }
        }
        
    def add_agent(self, agent_type: AgentType, agent: BaseAgent) -> None:
        """Add an agent to the coordinator."""
        if not isinstance(agent_type, AgentType):
            raise ValueError(f"Invalid agent type: {agent_type}")
        self.agents[agent_type] = agent

    async def process_message(self, user_input: str, model_choice: str = "claude-3-5-haiku") -> Dict[str, Any]:
        """Process an incoming user message and coordinate the appropriate agent response.
        
        This is the main entry point for processing messages in the system, used by both
        the chat interface and the orchestrator.
        
        Args:
            user_input: The user's input message
            model_choice: The selected model to use for processing (optional)
            
        Returns:
            Dict containing the response and any additional data
        """
        try:
            # Select appropriate agent based on input
            agent_type, confidence_agent, reasoning_agent, processing_mode = await self._select_agent(user_input, model_choice)
            
            # If confidence is too low, use TEXT_RESPONSE agent instead
            if confidence_agent <= 0.5:
                print(f"\n[Coordinator] Low confidence ({confidence_agent*100:.0f}%), falling back to TEXT_RESPONSE agent")
                print(f"[Coordinator] Original reasoning: {reasoning_agent}")
                agent_type = AgentType.TEXT_RESPONSE
                confidence_agent = 1.0  # Set high confidence for text response
                reasoning_agent = f"Falling back to general text response agent due to low confidence in specialized agents. Original reasoning: {reasoning_agent}"
            
            # Log the selected agent and confidence
            print(f"\n[Coordinator] Selected {agent_type.name} agent with {confidence_agent*100:.0f}% confidence")
            # print(f"[Coordinator] Reasoning: {reasoning_agent}")
            
            agent = self.agents.get(agent_type)
            if not agent:
                return {
                    "type": "error",
                    "content": f"No agent available for type: {agent_type.name}",
                    "metadata": {
                        "agent": "TEXT_RESPONSE",
                        "confidence": 0.0,
                        "reasoning": "Requested agent is not available in the system"
                    }
                }
                
            # Import socketio here to avoid circular import
            from core import socketio
            
            # Send start message
            socketio.emit('message', {
                'type': 'info',
                'content': f"🔄 Starting {agent_type.name.lower().replace('_', ' ')} task..."
            })

            # Get current molecule context
            from handlers.molecule_handler import get_current_molecule
            current_molecule = get_current_molecule()
            
            # Build context with molecule data
            context = {}
            if current_molecule:
                context['current_molecule'] = current_molecule
                context["processing_mode"] = processing_mode
                context["model_choice"] = model_choice
            print(f"[Coordinator] context: {context.keys()}")

            # Process the message with the selected agent and context
            response = await agent.process(user_input, context=context)
            # print(f"[Coordinator] Selected agent: {agent_type.name}, Confidence: {confidence_agent:.2f}")
            # print(f"[Coordinator] Agent response: {response}")

            # Handle tool responses
            if isinstance(response, dict) and response.get("type") == "tool_error":
                return {
                    "type": "error",
                    "content": response.get("content", "There is an error"),
                    "metadata": {
                        "agent": agent_type.name,
                        "confidence": response.get("confidence", 0.0),
                        "reasoning": response.get("reasoning", "No reasoning provided")
                    }
                }
            
            # Handle clarification responses
            if isinstance(response, dict) and response.get("type") == "clarification":
                return {
                    "type": "clarification",
                    "content": response.get("content", "Clarification needed"),
                    "metadata": {
                        "agent": agent_type.name,
                        "confidence": response.get("confidence", 0.0),
                        "reasoning": response.get("reasoning", "No reasoning provided")
                    }
                }
            
            # Preserve the original response type and structure for plot responses
            if isinstance(response, dict) and response.get("type") == "plot":
                response["metadata"] = {
                    "agent": agent_type.name,
                    "confidence": response.get("confidence", confidence_agent),
                    "reasoning": response.get("reasoning", reasoning_agent)
                }
                return response

            # Preserve the original response type and structure for plot responses
            if isinstance(response, dict) and response.get("type") == "molecule_plot":
                # print(f"[Coordinator] Response molecule_plot: {response}")
                response["metadata"] = {
                    "agent": agent_type.name,
                    "confidence": response.get("confidence", confidence_agent),
                    "reasoning": response.get("reasoning", reasoning_agent)
                }
                return response                
            
            # For non-dict responses, wrap them in a standard format
            return {
                "type": "text_response",
                "content": response,
                "metadata": {
                    "agent": agent_type.name,
                    "confidence": confidence_agent,
                    "reasoning": reasoning_agent
                }
            }
            
        except Exception as e:
            traceback.print_exc()
            return {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "metadata": {
                    "agent": "TEXT_RESPONSE",
                    "confidence": 0.0,
                    "reasoning": f"An error occurred while processing the request: {str(e)}"
                }
            }
            
    async def process(self, message: str, model_choice: str = None, context: Dict = None) -> Dict:
        """Legacy method for backward compatibility. Use process_message instead."""
        return await self.process_message(message, model_choice)

    def _create_agent_selection_prompt(self, message: str) -> str:
        """Create a prompt for the LLM to select an appropriate agent."""
        agent_info = json.dumps(self.agent_descriptions, indent=2)
        
        prompt = f"""IMPORTANT: You must respond with ONLY a JSON object. No other text or explanations.

        Task: Given the following user message and available agents, determine:
        1. The most appropriate agent to handle the request
        2. Whether to process a single molecule or batch of molecules

        User Message: "{message}"

        Available Agents:
        {agent_info}

        You must respond with EXACTLY this JSON format, nothing else:
        {{
            "agent_type": "MOLECULE_PLOT or NMR_PLOT or TEXT_RESPONSE or TOOL_USE or ORCHESTRATION or ANALYSIS",
            "confidence": <float between 0 and 1>,
            "processing_mode": "single" or "batch",
            "reasoning": "<one sentence explaining why this agent and mode were chosen. If confidence is low, explain why and suggest a clearer way to phrase the request>"
        }}

        For example, a valid response would be:
        {{
            "agent_type": "NMR_PLOT",
            "confidence": 0.95,
            "processing_mode": "single",
            "reasoning": "Request specifically mentions HSQC spectrum visualization for a specific molecule which is a core capability of the NMR Plot agent"
        }}

        Or for a batch processing example:
        {{
            "agent_type": "TOOL_USE",
            "confidence": 0.85,
            "processing_mode": "batch",
            "reasoning": "Request asks to calculate thresholds for all molecules in the dataset, requiring batch processing through the tool agent"
        }}

        Or for a low confidence example:
        {{
            "agent_type": "TEXT_RESPONSE",
            "confidence": 0.3,
            "reasoning": "Request is too vague to determine specific visualization needs - consider specifying if you want to see a molecule structure, NMR spectrum, or get information about a specific topic"
        }}
        """
        
        # print("\n[Coordinator] Generated prompt for agent selection:")
        # print("----------------------------------------")
        # print(prompt)
        # print("----------------------------------------")
        
        return prompt

    async def _select_agent(self, message: str, model_choice: str) -> Tuple[AgentType, float, str, str]:
        """
        Select the most appropriate agent based on message content.
        
        Args:
            message: The user message
            model_choice: The LLM model to use for processing
            
        Returns:
            Tuple of (AgentType, confidence_score, reasoning, processing_mode)
        """
        # Prepare the prompt for agent selection
        prompt = self._create_agent_selection_prompt(message)
        
        try:
            print("\n[Coordinator] Requesting agent selection from LLM...")
            # print(f"[Coordinator] Input message: {message}")
            print(f"[Coordinator] Using model: {model_choice}")
            
            # Get LLM response with JSON validation
            response = await self.llm_service.get_completion(
                prompt, 
                model=model_choice,
                require_json=True,  # Enable JSON validation
                max_retries=3  # Allow up to 3 retries
            )
            
            print("\n[Coordinator] Raw LLM response for agent selection:")
            # print("----------------------------------------")
            # # print(response)
            # print("----------------------------------------")
            
            # Handle error responses
            if response.startswith("Error in LLM completion:"):
                raise ValueError(response)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                print("\n[Coordinator] Parsed agent selection result:")
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError as e:
                print(f"[Coordinator] Invalid JSON response: {response}")
                raise ValueError(f"Failed to parse JSON response: {str(e)}")
            
            # Convert string agent type to enum
            agent_type_str = result["agent_type"].upper()  # Ensure uppercase
            if agent_type_str not in ["MOLECULE_PLOT", "NMR_PLOT", "TEXT_RESPONSE", "TOOL_USE", "ORCHESTRATION", "ANALYSIS"]:
                raise ValueError(f"Invalid agent type: {agent_type_str}")
                
            agent_type = AgentType[agent_type_str]
            confidence = float(result["confidence"])
            reasoning = str(result.get("reasoning", "No reasoning provided"))
            processing_mode = str(result.get("processing_mode", "single")).lower()
            
            if processing_mode not in ["single", "batch"]:
                processing_mode = "single"  # Default to single if invalid
            
            return agent_type, confidence, reasoning, processing_mode
            
        except Exception as e:
            print(f"[Coordinator] Error selecting agent: {str(e)}")
            return AgentType.TEXT_RESPONSE, 0.0, f"Error selecting agent: {str(e)}", "single"