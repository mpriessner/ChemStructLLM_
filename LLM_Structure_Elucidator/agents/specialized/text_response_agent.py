from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from services.llm_service import LLMService

class TextResponseAgent(BaseAgent):
    """Agent for handling general text-based responses."""
    def __init__(self, llm_service: LLMService):
        capabilities = [
            "Natural language understanding",
            "General question answering",
            "Contextual responses",
            "Information retrieval"
        ]
        super().__init__(name="Text Response", capabilities=capabilities)
        self.llm_service = llm_service

    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message and generate a text response.
        
        Args:
            message: The user message to process
            context: Additional context for processing (optional)
        """
        print("\n[Text Response Agent] ====== Starting Processing ======")
        print(f"[Text Response Agent] Message: {message}")
        print(f"[Text Response Agent] Context: {context}")
        model_choice = context.get('model_choice', 'gemini-flash')
        processing_mode = context.get('processing_mode', 'single')

        try:
            # Get response from LLM with proper system prompt
            print("[Text Response Agent] Getting response from LLM...")
            response = await self.llm_service.get_completion(
                message=message,
                model=model_choice,  
                system="You are an AI assistant specializing in chemical structure analysis and interpretation.",
                agent_name=self.name
            )
            
            # Handle error responses
            if response.startswith("Error in LLM completion:"):
                return {
                    "type": "error",
                    "content": {
                        "response": f"Error: {response}"
                    }
                }
            
            return response
            # return {
            #     "type": "text_response",
            #     "content": {
            #         "response": response
            #     }
            # }
            
        except Exception as e:
            error_msg = f"Error processing text response: {str(e)}"
            print(f"[Text Response Agent] {error_msg}")
            return {
                "type": "error",
                "content": {
                    "response": error_msg
                }
            }

    def can_handle(self, message):
        """Check if this agent should handle the message"""
        # This agent handles any text that doesn't match other specialized agents
        return True