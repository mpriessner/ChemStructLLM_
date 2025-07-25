"""
LLM API integration service.
"""
from typing import Dict, Any, Optional, List, Tuple
import anthropic
from anthropic import AsyncAnthropic
import openai
import google.generativeai as genai
from google import genai as genai_think
import json
import base64
from pathlib import Path
from config.settings import ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_AZURE_ENDPOINT, DEEPSEEK_AZURE_API_KEY, KIMI_API_KEY
import logging
import requests
import asyncio
import ast
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
# from openai import Client

# Initialize logger
logger = logging.getLogger(__name__)

class LLMService:
    """Handles interactions with LLM APIs through the agent system."""

    _instance = None

    # Model mapping dictionary
    MODEL_MAPPING = {
        # Claude models
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
        # GPT models
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o",
        # O3 models
        "o3-mini": "o3-mini",
        # Gemini models
        "gemini-flash": "models/gemini-1.5-flash",
        "gemini-pro": "models/gemini-pro",
        "gemini-thinking": "gemini-2.0-flash-thinking-exp-01-21",
        # DeepSeek models
        "deepseek-reasoner": "deepseek-reasoner",
        # Kimi models
        "kimi-thinking": "kimi-k1.5-preview"
    }

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    KIMI_BASE_URL = "https://api.moonshot.ai/v1"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            # Initialize Gemini models
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_pro = genai.GenerativeModel(self.MODEL_MAPPING["gemini-pro"])
            self.gemini_flash = genai.GenerativeModel(self.MODEL_MAPPING["gemini-flash"])
            # Thinking model uses Client
            self.gemini_thinking = genai_think.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
            # Initialize Kimi client
            # self.kimi_client = Client(api_key=KIMI_API_KEY, base_url=self.KIMI_BASE_URL)
            self._initialized = True
            self.conversation_history: List[Dict[str, Any]] = []

    async def get_completion(self, 
                           message: str, 
                           model: str = "claude-3-5-haiku",
                           context: Optional[Dict[str, Any]] = None,
                           agent_name: Optional[str] = None,
                           max_tokens: int = 2000,
                           system: Optional[str] = None,
                           require_json: bool = False,
                           max_retries: int = 3) -> Dict[str, str]:
        """Get completion from the specified LLM model through an agent.
        
        Args:
            message: The message to send to the LLM
            model: The model to use (e.g., "claude-3-5-haiku", "gemini-flash")
            context: Optional context dictionary
            agent_name: Optional agent name for conversation history
            max_tokens: Maximum tokens in response
            system: Optional system prompt
            require_json: If True, validates response is valid JSON and retries if not
            max_retries: Maximum number of retries for JSON validation
        """
        try:
            # Validate and normalize model name
            if model == "claude-3-haiku":  # Handle legacy model name
                model = "claude-3-5-haiku"

            if model not in self.MODEL_MAPPING:
                print(f"[LLM Service] Warning: Model {model} not found in mapping, falling back to claude-3-5-haiku")
                model = "claude-3-5-haiku"

            # Add message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "agent": agent_name
            })

            # Prepare context for the API call
            if context is None:
                context = {}
            if system:
                context["system"] = system

            # Add JSON formatting requirement if needed
            if require_json:
                print("\n[LLM Service] JSON response required")
                if system:
                    system = system + "\nIMPORTANT: You must respond with valid JSON only. No other text or explanation."
                else:
                    system = "IMPORTANT: You must respond with valid JSON only. No other text or explanation."
                context["system"] = system
                # print(f"[LLM Service] System prompt: {system}")

            # Prepare context for the API call
            full_context = self._prepare_context(context)
            print(f"\n[LLM Service] Sending request to model: {model}")

            # Try multiple times if JSON is required
            attempts = 0
            while attempts < max_retries:
                try:
                    # Get response based on model type
                    try:
                        if model.startswith("claude"):
                            # Map the model name to its full identifier
                            full_model_name = self.MODEL_MAPPING.get(model, model)
                            response = await self._get_anthropic_completion(message, full_model_name, full_context, max_tokens)
                        elif model.startswith("gpt") or model.startswith("o3"):
                            response = await self._get_openai_completion(message, model, full_context, max_tokens)
                        elif model.startswith("gemini"):
                            response = await self._get_gemini_completion(message, model, full_context, max_tokens)
                        # elif model == "deepseek-reasoner":
                            #     response = await self._query_deepseek(message, system)
                        elif model.startswith("deepseek"):
                            response = await self.query_deepseek_azure(message, system, max_tokens)
                        elif model.startswith("kimi"):
                            response = self._get_kimi_completion(message, model, max_tokens, system)
                        else:
                            raise ValueError(f"Unsupported model: {model}")
                    except Exception as e:
                        print(f"[LLM Service] Error during attempt {attempts + 1}: {str(e)}")
                        if attempts < max_retries - 1:
                            attempts += 1
                            continue
                        else:
                            raise

                    print(f"\n[LLM Service] Raw response from {model}:")
                    print("----------------------------------------")

                    # Validate JSON if required
                    if require_json:
                        response = response.strip()
                        try:
                            # Test if response is valid JSON
                            _ = json.loads(response)
                            print("[LLM Service] Successfully validated JSON response")
                            # If we get here, JSON is valid, break the loop
                            break
                        except json.JSONDecodeError as e:
                            print(f"[LLM Service] JSON validation failed: {str(e)}")
                            if attempts < max_retries - 1:
                                print(f"[LLM Service] Attempt {attempts + 1}: Invalid JSON response, retrying...")
                                message = message + "\nYour previous response was not valid JSON. Please provide ONLY a valid JSON object with no additional text."
                                attempts += 1
                                continue
                            else:
                                raise ValueError("Failed to get valid JSON response after all retries")
                    else:
                        # If JSON not required, just return the response
                        break

                except Exception as e:
                    print(f"[LLM Service] Error during attempt {attempts + 1}: {str(e)}")
                    if attempts < max_retries - 1:
                        attempts += 1
                        continue
                    else:
                        raise

            # Add response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "agent": agent_name,
                "model": model
            })

            # Ensure we return a string
            return response if isinstance(response, str) else str(response)

        except Exception as e:
            error_msg = f"Error in LLM completion: {str(e)}"
            print(f"[LLM Service] {error_msg}")
            # Return error message as string instead of dict
            return error_msg

    def _prepare_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare context for API calls."""
        base_context = {
            "conversation_history": self.conversation_history[-5:] if self.conversation_history else []
        }
        if context:
            base_context.update(context)
        return base_context

    async def _get_anthropic_completion(self, message: str, model: str, context: Dict[str, Any], max_tokens: int) -> str:
        """Get completion from Anthropic's Claude."""
        system_prompt = self._create_system_prompt(context)

        # print(f"\n[LLM Service] Sending request to Anthropic's Claude model: {model}")
        # print(f"[LLM Service] System prompt: {system_prompt}")
        # print(f"[LLM Service] User message: {message}")

        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": message}]
        )
        # print(f"\n[LLM Service] Raw response from Claude: {response.content[0].text}")
        # print("----------------------------------------")
        return response.content[0].text

    async def _get_openai_completion(self, message: str, model: str, context: Dict[str, Any], max_tokens: int) -> str:
        """Get completion from OpenAI's GPT or O3."""
        openai.api_key = OPENAI_API_KEY
        openai.api_base = "https://api.openai.com/v1"
        system_prompt = self._create_system_prompt(context)

        # Base parameters for all OpenAI models
        params = {
            "model": model,
            "max_tokens": 8000,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        }
        
        # Add O3-specific parameters
        if model.startswith("o3"):
            params = {
            "model": model,
            "max_completion_tokens": 20000,
            "reasoning_effort": "high",  ###low or medium or high
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        }

        response = openai.ChatCompletion.create(**params)
        return response.choices[0].message.content

    async def _get_gemini_completion(self, message: str, model: str, context: Dict[str, Any], max_tokens: int) -> str:
        """Get completion from Google's Gemini."""
        import ast
        
        system_prompt = self._create_system_prompt(context)
        
        # Combine system prompt and message
        full_prompt = f"{system_prompt}\n\nUser: {message}"

        try:
            if model == "gemini-thinking":
                # Use the specific client for the thinking model
                response = self.gemini_thinking.models.generate_content(
                    model=self.MODEL_MAPPING["gemini-thinking"],
                    contents=full_prompt
                )
                
                # For gemini-thinking, extract and parse JSON content
                raw_text = response.text
                return raw_text
                  
            else:
                # Get the appropriate model instance
                if model == "gemini-pro":
                    model_instance = self.gemini_pro
                elif model == "gemini-flash":
                    model_instance = self.gemini_flash
                else:
                    raise ValueError(f"Unknown Gemini model: {model}")

                # Generate content with the model
                response = model_instance.generate_content(
                    full_prompt,
                    generation_config={"max_output_tokens": 20000}
                )

                # Get response text
                text = response.text
                if text.startswith("```json"):
                    text = text[7:]  # Remove ```json prefix
                text = text.replace("```", "").strip()
                
                return text

        except Exception as e:
            logger.error(f"Error in Gemini completion: {str(e)}")
            raise

    # async def _query_deepseek(self, message: str, system: str = None) -> Dict[str, str]:
        #     """Query the DeepSeek API using the OpenAI library format.
        
    #     Returns:
    #         Dict containing 'content' and 'reasoning_content' from the DeepSeek response
    #     """
    #     try:
    #         # Configure OpenAI client for DeepSeek
    #         openai.api_base = "https://api.deepseek.com/v1"
    #         openai.api_key = DEEPSEEK_API_KEY

    #         # Prepare messages
    #         messages = []
    #         if system:
    #             messages.append({"role": "system", "content": system})
    #         messages.append({"role": "user", "content": message})

    #         # Create chat completion request using non-async method
    #         response = openai.ChatCompletion.create(
    #             model="deepseek-reasoner",
    #             messages=messages,
    #             stream=False,
    #             temperature=0.3
    #         )

    #         # Extract response content and reasoning
    #         message = response.choices[0].message
    #         return {
    #             'content': message['content'],
    #             'reasoning_content': message.get('reasoning_content', '')  # Get reasoning_content if available
    #         }

    #     except openai.error.AuthenticationError:
    #         logger.error("Authentication failed. Please check your DeepSeek API key.")
    #         raise Exception("Authentication failed")
    #     except openai.error.RateLimitError:
    #         logger.error("Rate limit exceeded. Please try again later.")
    #         raise Exception("Rate limit exceeded")
    #     except Exception as e:
    #         logger.error(f"Error querying DeepSeek API: {str(e)}")
    #         raise

    def _extract_parts(self, content: str) -> tuple:
        """Extract thinking and content parts from a string response.
        
        Args:
            content: String containing the full response text
            
        Returns:
            tuple: (thinking, content) where thinking is the text between <think> tags
                  and content is everything after </think> or the full text if no tags
        """
        # Extract thinking part (between <think> and </think>)
        think_start = content.find('<think>') + len('<think>')
        think_end = content.find('</think>')
        thinking = content[think_start:think_end].strip() if '<think>' in content else ''
        
        # Extract content part (everything after </think>)
        content = content[think_end + len('</think>'):].strip() if '</think>' in content else content
        
        return thinking, content

    def _extract_parts_kimi(self, content: str) -> tuple:
        """Extract thinking and content parts from a Kimi response string.
        
        Args:
            content: String containing the full response text from Kimi
            
        Returns:
            tuple: (thinking, content) where thinking is the text between ◀think▷ tags
                and content is everything after ◀/think▷ or the full text if no tags
        """
        # Define Kimi's specific tags
        think_start_tag = '\u25c1think\u25b7'
        think_end_tag = '\u25c1/think\u25b7'
        
        # Extract thinking part (between ◀think▷ and ◀/think▷)
        think_start = content.find(think_start_tag) + len(think_start_tag)
        think_end = content.find(think_end_tag)
        thinking = content[think_start:think_end].strip() if think_start_tag in content else ''
        
        # Extract content part (everything after ◀/think▷)
        content = content[think_end + len(think_end_tag):].strip() if think_end_tag in content else content
        
        return thinking, content
      
    def _get_kimi_completion(self, message: str, model: str, max_tokens: int, system: str = None, stream: bool = False) -> Tuple[str, str]:
        """Get completion from Kimi's thinking model with optional streaming.
        Args:
            message: The input message
            model: Model identifier
            max_tokens: Maximum tokens in response
            system: Optional system prompt
            stream: Whether to use streaming mode (default: False)
        Returns:
            Tuple[str, str]: (thinking, content) where thinking is the extracted thinking part
                           and content is the main response content
        """
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})
            
            openai.api_key = "sk-QWUjltc5N6CWT4FOfjIPl0X8ZhotO88TY0Yk3ncl6iwir222"
            openai.api_base = "https://api.moonshot.ai/v1"

            response = openai.ChatCompletion.create(
                model=self.MODEL_MAPPING[model],
                messages=messages,
                temperature=0.3,
                stream=stream,
                max_tokens=8000,
            )

            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                response_text = full_response
            else:
                # Handle non-streaming response
                response_text = response.choices[0].message.content

            logger.info(f"Response from Kimi: {response_text}")
            # Extract thinking and content parts
            thinking, content = self._extract_parts_kimi(response_text)
            return thinking, content

        except Exception as e:
            logger.error(f"Error in Kimi completion: {str(e)}")
            raise

    async def query_deepseek_azure(self, message: str, system: str = None, max_tokens: int = 500) -> Dict[str, str]:
        """Query the DeepSeek API using Azure AI inference with streaming and exception handling."""
        try:
            client = ChatCompletionsClient(
                endpoint=DEEPSEEK_AZURE_ENDPOINT,
                credential=AzureKeyCredential(DEEPSEEK_AZURE_API_KEY),
            )

            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(UserMessage(content=message))

            # Initialize an empty string to accumulate the response
            full_response = ""

            # Stream response
            response = client.complete(
                model="DeepSeek-R1",
                messages=messages,
                stream=True,  # Enable streaming
                max_tokens=8000,  # Use the provided max_tokens parameter
            )

            # Iterate over the streamed response and accumulate the content
            for update in response:
                try:
                    content = update.choices[0].delta.content or ""
                    full_response += content
                    print(content, end="", flush=True)
                except Exception as e:
                    logger.error(f"Error processing streamed content: {str(e)}")
                    break  # Exit the loop on error

            # After the loop, full_response contains the complete response
            thinking, content = self._extract_parts(full_response)
            return thinking, content 


        except Exception as e:
            logger.error(f"Error querying DeepSeek Azure API: {str(e)}")
            return {"error": str(e)}

    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create system prompt based on context."""
        if context.get("system"):
            base_prompt = context["system"]
        else:
            base_prompt = "You are an AI assistant specializing in chemical structure analysis and interpretation."

        if context.get("conversation_history"):
            base_prompt += "\nPrevious conversation context is provided for reference."

        return base_prompt

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    async def analyze_with_vision(self, 
                                prompt: str,
                                image_path: str,
                                model: str = "claude-3-5-sonnet",
                                max_tokens: int = 2048,
                                system: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image using LLM vision capabilities.
        
        Args:
            prompt: The prompt describing what to analyze in the image
            image_path: Path to the image file
            model: The model to use (default: claude-3-5-sonnet for best vision capabilities)
            max_tokens: Maximum tokens in response
            system: Optional system prompt
            
        Returns:
            The LLM's analysis of the image
        """
        try:
            # Validate and normalize model name
            if model == "claude-3-sonnet":  # Handle legacy model name
                model = "claude-3-5-sonnet"
            
            # For vision tasks, we only support Claude models currently
            if not model.startswith("claude"):
                logger.warning(f"Model {model} not supported for vision tasks, falling back to claude-3-5-sonnet")
                model = "claude-3-5-sonnet"
            
            # Get the full model name from mapping
            full_model_name = self.MODEL_MAPPING.get(model, model)
            
            # Read image file
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as f:
                image_data = f.read()

            # Create content array with image and text
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_data).decode()
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            # Set up system prompt
            if not system:
                system = ("You are an expert in chemical structure analysis and NMR spectroscopy. "
                         "Analyze the provided molecular structure image and provide detailed insights ")

            # Prepare the request body
            request_body = {
                "model": full_model_name,  # Use mapped model name if available, otherwise use as-is
                "max_tokens": 8000,
                "system": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }

            # Set up headers with API key, API version and beta features
            headers = {
                "x-api-key": self.anthropic_client.api_key,  # Use API key from initialized client
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "content-type": "application/json"
            }

            # Make the API call with custom headers
            response = await self.anthropic_client._client.post(
                "https://api.anthropic.com/v1/messages",
                json=request_body,
                headers=headers
            )

            # Log the response status
            logger.info(f"Vision API response status: {response.status_code}")

            # Handle non-200 responses
            if response.status_code != 200:
                error_msg = f"Vision API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'analysis_text': error_msg,
                    'confidence': 0.0,
                    'structural_matches': [],
                    'mismatches': []
                }

            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON response: {str(e)}"
                logger.error(error_msg)
                return {
                    'analysis_text': error_msg,
                    'confidence': 0.0,
                    'structural_matches': [],
                    'mismatches': []
                }

            # Log usage statistics if available
            try:
                if 'usage' in response_data:
                    print("\nUsage statistics for Vision Analysis:")
                    print(f"Input tokens: {response_data['usage'].get('input_tokens', 'N/A')}")
                    print(f"Output tokens: {response_data['usage'].get('output_tokens', 'N/A')}")
                    print(f"Cache creation input tokens: {response_data['usage'].get('cache_creation_input_tokens', 0)}")
                    print(f"Cache read input tokens: {response_data['usage'].get('cache_read_input_tokens', 0)}")
                    print("\n" + "-"*50 + "\n")
                else: 
                    logger.warning("No usage statistics in response")
            except Exception as e:
                logger.warning(f"Error logging usage statistics: {str(e)}")

            # Extract the analysis text with better error handling
            try:
                if 'content' in response_data and len(response_data['content']) > 0:
                    content = response_data['content'][0]
                    analysis_text = content['text'] if content['type'] == 'text' else str(content)
                    
                    # Return structured response
                    structured_response = {
                        'analysis_text': analysis_text,
                    }
                    
                    return structured_response
                else:
                    error_msg = "No content in response"
                    logger.error(error_msg)
                    return {
                        'analysis_text': error_msg,
                    }
            except Exception as e:
                error_msg = f"Error extracting content: {str(e)}"
                logger.error(error_msg)
                return {
                    'analysis_text': error_msg,
                }

        except Exception as e:
            error_msg = f"Error in vision analysis: {str(e)}"
            logger.error(error_msg)