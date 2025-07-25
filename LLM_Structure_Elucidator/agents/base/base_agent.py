"""
Base agent class and common utilities for the multi-agent system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.memory: Dict[str, Any] = {}
    
    @abstractmethod
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message and return a response.
        
        Args:
            message: The input message to process
            context: Optional context information
            
        Returns:
            Dict containing the response with at least 'type' and 'content' keys
        """
        pass
    
    def can_handle(self, task: str) -> bool:
        """Check if the agent can handle a specific task."""
        return any(capability.lower() in task.lower() for capability in self.capabilities)
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent's memory with new information."""
        self.memory[key] = value
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent's memory."""
        return self.memory.get(key)
    
    def __str__(self) -> str:
        return f"{self.name} Agent (Capabilities: {', '.join(self.capabilities)})"
