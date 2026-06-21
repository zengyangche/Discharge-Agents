"""
LLM client: unified interface for calling different models
"""
from typing import Dict, Any, Optional, List
import logging
import sys
import json
from pathlib import Path

# Add project root to path to ensure config module can be imported
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.config import (MODELS, MODEL_ZOO, OPENAI_API_KEY, OPENAI_API_BASE_URL)
    # If MODELS is empty, use MODEL_ZOO
    if not MODELS and MODEL_ZOO:
        MODELS = MODEL_ZOO
except ImportError:
    # If import from within utils directory fails, try direct import
    try:
        from utils.config import (MODELS, MODEL_ZOO, OPENAI_API_KEY, OPENAI_API_BASE_URL)
        # If MODELS is empty, use MODEL_ZOO
        if not MODELS and MODEL_ZOO:
            MODELS = MODEL_ZOO
    except ImportError:
        # If all fail, use default value
        MODELS = {}
        OPENAI_API_KEY = None
        OPENAI_API_BASE_URL = "https://api.openai.com/v1"

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting multiple models"""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize per-model clients (unified API platform)"""
        # Unified OpenAI-compatible API; all models go through the same client
        if OPENAI_API_KEY:
            try:
                import openai
                # Use the unified API base URL (integrated platform)
                self.clients["openai"] = openai.OpenAI(
                        api_key=OPENAI_API_KEY,
                        base_url=OPENAI_API_BASE_URL
                    )
                logger.info(f"Unified API client initialized, Base URL: {OPENAI_API_BASE_URL}")
            except ImportError:
                logger.warning("OpenAI library not installed, run: pip install openai")
            except Exception as e:
                logger.error(f"Failed to initialize API client: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    
    def generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text.
        
        Args:
            model_name: model name (e.g. "gpt-4o")
            prompt: user prompt
            system_prompt: system prompt
            temperature: sampling temperature (None uses model config)
            max_tokens: max output tokens (None uses model config)
            min_tokens: min output tokens (None uses model config, or omitted if unset)
            **kwargs: extra parameters (frequency_penalty, presence_penalty, etc.)
            
        Returns:
            generated text
        """
        model_config = MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        provider = model_config["provider"]
        
        # Get default parameters from model config; caller-provided values take priority
        final_temperature = temperature if temperature is not None else model_config.get("temperature", 0.7)
        final_max_tokens = max_tokens if max_tokens is not None else model_config.get("max_tokens", 2000)
        final_min_tokens = min_tokens if min_tokens is not None else model_config.get("min_tokens", None)
        
        # Some models do not support frequency_penalty / presence_penalty (e.g., grok series)
        _NO_PENALTY_MODELS = ("grok",)
        _model_id = model_config.get("model_name", model_name).lower()
        _supports_penalty = not any(_model_id.startswith(p) for p in _NO_PENALTY_MODELS)

        # Merge other parameters from model config (e.g., frequency_penalty, presence_penalty)
        # If these parameters are already in kwargs, kwargs values take priority
        merged_kwargs = {}
        if _supports_penalty:
            if "frequency_penalty" in model_config:
                merged_kwargs["frequency_penalty"] = kwargs.get("frequency_penalty", model_config["frequency_penalty"])
            if "presence_penalty" in model_config:
                merged_kwargs["presence_penalty"] = kwargs.get("presence_penalty", model_config["presence_penalty"])
        # Add other parameters from kwargs
        merged_kwargs.update({k: v for k, v in kwargs.items() if k not in ["frequency_penalty", "presence_penalty"]})
        
        # Unified OpenAI-compatible API; all models call through _generate_openai
        # Different models are distinguished by model name
        logger.info(f"LLM call: model={model_name}, max_tokens={final_max_tokens}, temperature={final_temperature}")
        return self._generate_openai(
                model_config["model_name"],
                prompt,
                system_prompt,
            final_temperature,
            final_max_tokens,
            min_tokens=final_min_tokens,
            **merged_kwargs
        )
    
    def _generate_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        min_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """OpenAI model generation"""
        if "openai" not in self.clients:
            raise ValueError(
                "OpenAI client not initialized. Please check:\n"
                "1. OPENAI_API_KEY is correctly configured\n"
                "2. openai library is installed (pip install openai)\n"
                "3. No errors occurred during initialization"
            )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build API call parameters
        # Note: OpenAI API does not support the min_tokens parameter, so it is not passed
        api_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add other parameters (exclude min_tokens, as OpenAI API does not support it)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "min_tokens"}
        api_kwargs.update(filtered_kwargs)

        # Gemini thinking model uses all token budget for reasoning, causing text output truncation.
        # Set thinking_budget to 0 via extra_body to disable thinking mode.
        if model.lower().startswith("gemini"):
            api_kwargs["extra_body"] = {"thinking": {"thinking_budget": 0}}

        # Log API call parameters (for debugging)
        logger.info(f"API call: model={model}, max_tokens={max_tokens}, temperature={temperature}")
        
        response = self.clients["openai"].chat.completions.create(**api_kwargs)
        
        # Log returned token count (if available)
        completion_tokens = None
        if hasattr(response, 'usage') and response.usage:
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            logger.info(f"API response: completion_tokens={completion_tokens}, total_tokens={total_tokens}, max_tokens limit={max_tokens}")
            if completion_tokens > max_tokens:
                logger.warning(f"Warning: generated token count ({completion_tokens}) exceeded max_tokens limit ({max_tokens})")
        
        # For Gemini models, if completion_tokens>0 but content is empty, log raw response (for debugging)
        if model.startswith('gemini') and completion_tokens and completion_tokens > 0:
            try:
                # Try to get raw response
                if hasattr(response, 'model_dump'):
                    response_dict = response.model_dump()
                    if response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        if hasattr(choice, 'message') and (not choice.message.content or len(choice.message.content) == 0):
                            logger.warning(f"Gemini model {model} returned empty content but completion_tokens={completion_tokens}, logging full response for debugging")
                            # Print full response dict for debugging (WARNING level)
                            try:
                                full_response_str = json.dumps(response_dict, indent=2, ensure_ascii=False, default=str)
                                # If response is too long, only print first 5000 characters
                                if len(full_response_str) > 5000:
                                    logger.warning(f"Full response dict (first 5000 chars): {full_response_str[:5000]}...")
                                else:
                                    logger.warning(f"Full response dict: {full_response_str}")
                            except Exception as e:
                                logger.warning(f"Unable to serialize response dict: {str(e)}")
                            
                            # Try to extract content directly from response dict
                            if 'choices' in response_dict and len(response_dict['choices']) > 0:
                                choice_dict = response_dict['choices'][0]
                                    logger.warning(f"Choice dict keys: {list(choice_dict.keys())}")
                                    if 'message' in choice_dict:
                                        msg_dict = choice_dict['message']
                                        logger.warning(f"Message dict keys: {list(msg_dict.keys())}")
                                    # Print all fields in message dict
                                        try:
                                            msg_str = json.dumps(msg_dict, indent=2, ensure_ascii=False, default=str)
                                            logger.warning(f"Full message dict contents: {msg_str}")
                                        except Exception as e:
                                            logger.warning(f"Unable to serialize message dict: {str(e)}")
                                            logger.warning(f"Raw message dict: {msg_dict}")
            except Exception as e:
                logger.warning(f"Unable to log full response: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Extract response content
        if not response.choices or len(response.choices) == 0:
            logger.error(f"API response has no choices, response object: {response}")
            return ""
        
        # Try multiple methods to extract content (compatible with different response formats)
        choice = response.choices[0]
        message_content = None
        
        # Method 1: Standard format response.choices[0].message.content
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            message_content = choice.message.content
        
        # Method 2: Check refusal field (Gemini models may use this field)
        if (message_content is None or message_content == "") and hasattr(choice, 'message'):
            message = choice.message
            if hasattr(message, 'refusal') and message.refusal:
                # Non-empty refusal means the model rejected the request
                logger.warning(f"Model {model} rejected the request, refusal: {message.refusal}")
                message_content = f"[Refusal: {message.refusal}]"
            elif hasattr(message, 'refusal') and message.refusal is None and message_content == "":
                # refusal is None but content is empty, possibly a Gemini special case
                # Try to extract from the raw response
                logger.warning(f"Model {model} returned empty content but completion_tokens>0, possibly a response format issue")
        
        # Method 3: Access choice.content directly (some API variants)
        if (message_content is None or message_content == "") and hasattr(choice, 'content'):
            message_content = choice.content
            logger.info(f"Extracted content via choice.content, model: {model}")
        
        # Method 4: Check if message has other fields
        if (message_content is None or message_content == "") and hasattr(choice, 'message'):
            message = choice.message
            # Try text field (some API variants)
            if hasattr(message, 'text'):
                message_content = message.text
                logger.info(f"Extracted content via message.text, model: {model}")
        
        # Method 5: Try to extract from raw response JSON (last resort)
        if (message_content is None or message_content == "") and hasattr(response, 'model_dump'):
            try:
                response_dict = response.model_dump()
                logger.warning(f"Attempting to extract from response dict, model: {model}")
                logger.warning(f"Response dict keys: {list(response_dict.keys())}")
                # Try to extract from dict
                if 'choices' in response_dict and len(response_dict['choices']) > 0:
                    choice_dict = response_dict['choices'][0]
                    logger.warning(f"choice dict keys: {list(choice_dict.keys())}")
                    logger.warning(f"Full choice dict: {json.dumps(choice_dict, indent=2, ensure_ascii=False, default=str)}")
                    if 'message' in choice_dict:
                        msg_dict = choice_dict['message']
                        logger.warning(f"message dict keys: {list(msg_dict.keys())}")
                        logger.warning(f"Full message dict: {json.dumps(msg_dict, indent=2, ensure_ascii=False, default=str)}")
                        # Try all possible fields
                        for key in ['content', 'text', 'message', 'response', 'parts', 'candidates']:
                            if key in msg_dict and msg_dict[key]:
                                if isinstance(msg_dict[key], str) and msg_dict[key].strip():
                                                                    message_content = str(msg_dict[key])
                                                                    logger.warning(f"✓ Content extracted from response dict, field: {key}, model: {model}, length: {len(message_content)}")
                                    break
                                elif isinstance(msg_dict[key], list) and len(msg_dict[key]) > 0:
                                    # If list, try first element
                                    first_item = msg_dict[key][0]
                                    if isinstance(first_item, dict):
                                        # Try to extract text or content from dict
                                        for sub_key in ['text', 'content', 'part', 'message']:
                                            if sub_key in first_item and first_item[sub_key]:
                                                if isinstance(first_item[sub_key], str):
                                                    message_content = first_item[sub_key]
                                                    logger.warning(f"✓ Content extracted from response dict list at {key}[0].{sub_key}, model: {model}, length: {len(message_content)}")
                                                    break
                                                elif isinstance(first_item[sub_key], dict) and 'text' in first_item[sub_key]:
                                                    message_content = first_item[sub_key]['text']
                                                    logger.warning(f"✓ Content extracted from response dict list at {key}[0].{sub_key}.text, model: {model}, length: {len(message_content)}")
                                                    break
                                        if message_content:
                                            break
                                    elif isinstance(first_item, str):
                                        message_content = first_item
                                        logger.warning(f"✓ Content extracted from response dict list, model: {model}, length: {len(message_content)}")
                                        break
                    # If not in message, try extracting directly from choice
                    if not message_content:
                        for key in ['content', 'text', 'delta']:
                            if key in choice_dict and choice_dict[key]:
                                if isinstance(choice_dict[key], str) and choice_dict[key].strip():
                                    message_content = choice_dict[key]
                                    logger.warning(f"✓ Content extracted from choice dict, field: {key}, model: {model}, length: {len(message_content)}")
                                    break
            except Exception as e:
                logger.warning(f"Unable to extract content from response dict: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Check if content is empty
        if message_content is None:
            logger.error(f"API returned None content, model: {model}")
            logger.error(f"Response structure: choices count={len(response.choices)}, choice type={type(choice)}")
            logger.error(f"choice attributes: {[attr for attr in dir(choice) if not attr.startswith('_')]}")
            if hasattr(choice, 'message'):
                logger.error(f"message attributes: {[attr for attr in dir(choice.message) if not attr.startswith('_')]}")
            return ""
        
        # Log content length (for debugging)
        if len(message_content) == 0:
            logger.warning(f"API returned empty string content, model: {model}, completion_tokens={completion_tokens if hasattr(response, 'usage') and response.usage else 'N/A'}")
            logger.warning(f"Response choices[0] type: {type(choice)}")
            logger.warning(f"Response message type: {type(choice.message) if hasattr(choice, 'message') else 'N/A'}")
            # Print full choice object (for debugging)
            try:
                choice_dict = {
                    'type': str(type(choice)),
                    'message_type': str(type(choice.message)) if hasattr(choice, 'message') else None,
                    'message_content': str(choice.message.content) if hasattr(choice, 'message') and hasattr(choice.message, 'content') else None,
                    'message_dict': choice.message.model_dump() if hasattr(choice.message, 'model_dump') else None
                }
                logger.warning(f"Choice object details: {json.dumps(choice_dict, indent=2, ensure_ascii=False)}")
            except Exception as e:
                logger.warning(f"Unable to serialize choice object: {str(e)}")
        
        # Detect truncation: finish_reason == "length" indicates output was cut off due to token exhaustion
        if response.choices and response.choices[0].finish_reason == "length":
            logger.warning(
                f"Model {model} output truncated due to max_tokens exhaustion "
                f"(completion_tokens={completion_tokens}), [TRUNCATED] appended to content"
            )
            message_content = (message_content or "") + " [TRUNCATED]"

        return message_content
    
    # Note: _generate_qwen, _generate_deepseek, _generate_gemini have been removed
    # All models are now called via _generate_openai (using unified integrated platform API)
    
    def batch_generate(
        self,
        model_names: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Batch-generate using multiple models.
        
        Returns:
            {model_name: generated_text}
        """
        results = {}
        for model_name in model_names:
            try:
                results[model_name] = self.generate(
                    model_name, prompt, system_prompt, **kwargs
                )
            except Exception as e:
                logger.error(f"Model {model_name} generation failed: {str(e)}")
                results[model_name] = None
        return results
