from typing import Optional, List, Tuple, Dict
from llm_clients.openai import OpenAIClient
from llm_clients.claude import ClaudeClient
from llm_clients.gemini import GeminiClient
from llm_clients.grok import GrokClient
from enum import Enum


class LLMProvider(Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GROK = "grok"
    GEMINI = "gemini"

class LLMModel(Enum):
    """Enum for supported LLM models."""

    # OpenAI models
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest"
    O1_MINI = "o1-mini"
    O1 = "o1"
    O3_MINI = "o3-mini"
    O3 = "o3"
    O4_MINI = "o4-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_5_CHAT_LATEST = "gpt-5-chat-latest"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"

    # Anthropic Claude models
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250805"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # xAI / Grok models
    GROK_2_LATEST = "grok-2-latest"
    GROK_2_1212 = "grok-2-1212"
    GROK_2_VISION_1212 = "grok-2-vision-1212"
    GROK_3 = "grok-3"
    GROK_3_MINI = "grok-3-mini"
    GROK_4_0709 = "grok-4-0709"
    GROK_4_FAST_NON_REASONING = "grok-4-fast-non-reasoning"
    GROK_4_FAST_REASONING = "grok-4-fast-reasoning"
    GROK_CODE_FAST_1 = "grok-code-fast-1"

    # Google Gemini models
    GEMINI_3_PRO = "gemini-3-pro-preview"
    GEMINI_3_PRO_IMAGE = "gemini-3-pro-image-preview"
    GEMINI_3_FLASH = "gemini-3-flash-preview"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview-09-2025"
    GEMINI_2_5_FLASH_IMAGE = "gemini-2.5-flash-image"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH_LITE_PREVIEW = "gemini-2.5-flash-lite-preview-09-2025"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"

class LLMCaller:
    """Unified client for making LLM calls across different providers."""
    
    def __init__(self):
        self.clients = {
            LLMProvider.OPENAI: OpenAIClient(),
            LLMProvider.CLAUDE: ClaudeClient(),
            LLMProvider.GROK: GrokClient(),
            LLMProvider.GEMINI: GeminiClient()
        }
        
        # Model mapping: which models are supported by which providers
        self.model_provider_mapping = {
            # OpenAI models
            LLMModel.GPT_4: [LLMProvider.OPENAI],
            LLMModel.GPT_3_5_TURBO: [LLMProvider.OPENAI],
            LLMModel.GPT_4_TURBO: [LLMProvider.OPENAI],
            LLMModel.GPT_4O: [LLMProvider.OPENAI],
            LLMModel.GPT_4O_MINI: [LLMProvider.OPENAI],
            LLMModel.CHATGPT_4O_LATEST: [LLMProvider.OPENAI],
            LLMModel.O1_MINI: [LLMProvider.OPENAI],
            LLMModel.O1: [LLMProvider.OPENAI],
            LLMModel.O3_MINI: [LLMProvider.OPENAI],
            LLMModel.O3: [LLMProvider.OPENAI],
            LLMModel.O4_MINI: [LLMProvider.OPENAI],
            LLMModel.GPT_4_1: [LLMProvider.OPENAI],
            LLMModel.GPT_4_1_MINI: [LLMProvider.OPENAI],
            LLMModel.GPT_4_1_NANO: [LLMProvider.OPENAI],
            LLMModel.GPT_5_CHAT_LATEST: [LLMProvider.OPENAI],
            LLMModel.GPT_5: [LLMProvider.OPENAI],
            LLMModel.GPT_5_MINI: [LLMProvider.OPENAI],
            LLMModel.GPT_5_NANO: [LLMProvider.OPENAI],
            
            # Claude models
            LLMModel.CLAUDE_HAIKU_4_5: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_SONNET_4_5: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_OPUS_4_1: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_OPUS_4: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_SONNET_4: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_3_7_SONNET: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_3_5_HAIKU: [LLMProvider.CLAUDE],
            LLMModel.CLAUDE_3_HAIKU: [LLMProvider.CLAUDE],
                        
            # Grok models
            LLMModel.GROK_2_LATEST: [LLMProvider.GROK],
            LLMModel.GROK_2_1212: [LLMProvider.GROK],
            LLMModel.GROK_2_VISION_1212: [LLMProvider.GROK],
            LLMModel.GROK_3: [LLMProvider.GROK],
            LLMModel.GROK_3_MINI: [LLMProvider.GROK],
            LLMModel.GROK_4_0709: [LLMProvider.GROK],
            LLMModel.GROK_4_FAST_NON_REASONING: [LLMProvider.GROK],
            LLMModel.GROK_4_FAST_REASONING: [LLMProvider.GROK],
            LLMModel.GROK_CODE_FAST_1: [LLMProvider.GROK],
            
            # Gemini models
            LLMModel.GEMINI_3_PRO: [LLMProvider.GEMINI],
            LLMModel.GEMINI_3_PRO_IMAGE: [LLMProvider.GEMINI],
            LLMModel.GEMINI_3_FLASH: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_5_FLASH: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_5_FLASH_PREVIEW: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_5_FLASH_IMAGE: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_5_FLASH_LITE: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_5_FLASH_LITE_PREVIEW: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_5_PRO: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_0_FLASH: [LLMProvider.GEMINI],
            LLMModel.GEMINI_2_0_FLASH_LITE: [LLMProvider.GEMINI],
        }
    
    def call_model(
        self,
        model_configs: List[Tuple[LLMProvider, LLMModel]],
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None
    ) -> str:
        """
        Call LLM models with the given configuration.
        
        Args:
            model_configs: List of (provider, model) tuples to try
            prompt: User prompt
            system_message: Optional system message
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            message_history: Optional message history
            images: Optional list of base64-encoded images to include in the request
            
        Returns:
            str: Response text from the first successful model
            
        Raises:
            Exception: If all models fail
        """
        if not model_configs:
            raise Exception("No model configurations provided")

        errors = ""
                
        for _, (provider, model_enum) in enumerate(model_configs):
            try:
                # Validate model configuration
                if model_enum not in self.model_provider_mapping:
                    continue
                
                if provider not in self.model_provider_mapping[model_enum]:
                    continue

                print(f"Calling model {model_enum.value} with provider {provider}")
                
                client = self.clients[provider]

                response_text = client.simple_chat(
                    model=model_enum.value,
                    prompt=prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    message_history=message_history,
                    images=images
                )

                if not response_text:
                    errors += f"Model {model_enum.value} failed\n"
                    continue
                
                return response_text
            except Exception as e:
                errors += f"Model {model_enum.value} failed: {str(e)}\n"
                continue
        
        raise Exception(f"All models failed: {errors}")
