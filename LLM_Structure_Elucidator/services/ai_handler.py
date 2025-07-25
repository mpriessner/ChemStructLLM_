"""
AI Handler for managing LLM API calls.
"""
from typing import Dict, Any, Optional
from services.llm_service import LLMService

class AIHandler:
    def __init__(self):
        self.llm_service = LLMService()

    def make_api_call(self, model: str, message: str, system: Optional[str] = None) -> str:
        """Make an API call to the LLM service."""
        try:
            print(f"[AI Handler] Making API call with model: {model}")
            print(f"[AI Handler] Message: {message}")
            if system:
                print(f"[AI Handler] System prompt: {system}")

            # Create context with system prompt if provided
            context = {"system_prompt": system} if system else None

            response = self.llm_service.get_completion(
                message=message,
                model=model,
                context=context
            )
            
            if isinstance(response, dict):
                return response.get("content", "")
            return response

        except Exception as e:
            print(f"[AI Handler] Error in API call: {str(e)}")
            raise

# Global instance
ai_handler = AIHandler()
