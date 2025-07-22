import os
import asyncio
import logging
import httpx
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Tuple, Coroutine
from abc import ABC, abstractmethod
from datetime import datetime
# Core imports
import openai
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

# LangChain imports (following user requirements)
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain packages not found. Some features will be disabled.")

# LangSmith imports (following user requirements)
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith not available. Tracing will be disabled.")

# Claude/Anthropic imports
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logging.warning("Anthropic package not found. Claude models will be unavailable.")

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google GenerativeAI SDK not found. Gemini models will be unavailable.")

# Groq imports
try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq Python SDK not found. Groq models will be unavailable.")

# Local Llama imports
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("llama-cpp-python not found. Local Llama models will be unavailable.")

# OpenRouter (uses OpenAI client format)
OPENROUTER_AVAILABLE = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration class for LLM Manager"""
    
    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    
    # Model settings
    default_model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 2000
    
    # OpenRouter settings
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    
    # Local Llama settings
    llama_model_path: str = field(default_factory=lambda: os.getenv("LLAMA_MODEL_PATH", ""))
    
    # Timeout settings
    timeout_seconds: float = 180.0
    max_retries: int = 2
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # System prompts
    default_system_prompt: str = """You are Insight Agent, a precision AI assistant. Your primary mission is to provide accurate, clear, and trustworthy answers based **exclusively** on the information provided to you in the 'CONTEXT' section.
    f"🕒 **Current Time:** {formatted_time}\n\n"
    f"📘 **CONTEXT:**\n{context_str}\n\n"

---
### 🚨 **Core Directives: Non-Negotiable Rules**
---

1.  **ABSOLUTE CONTEXT ADHERENCE:** Your entire response MUST be derived from the provided context. NEVER use your own general knowledge, make assumptions, or fill in gaps. If the context does not contain the answer, you MUST state it directly using the specified protocol below. This is your most important rule.

2.  **MANDATORY STARTING FORMAT:** Every single response you generate MUST begin with an emoji followed by a bolded headline.
    *   **Format:** `🏷️ **Your Concise Headline Here**`
    *   **Example:** `📊 **Quarterly Report Summary**`

3.  **UNFAILING CITATIONS:** Every piece of information, every fact, and every number you present MUST be immediately followed by its source citation. There are no exceptions.
    *   **For Knowledge Base or User Documents:** Cite using the document's name.
        *   *Example:* "The project's budget was $50,000 [Source: project_plan.pdf]."
    *   **For Web Search Results:** Cite using the numbered index `[1]`, `[2]`, etc. The full URLs corresponding to these numbers will be provided to the user automatically.
        *   *Example:* "The new model was released last week [1]."
    *   **Multiple Sources:** If a sentence synthesizes information from multiple sources, cite all of them.
        *   *Example:* "The market is expected to grow by 10% [1, Source: market_analysis.docx]."

4.  **INSUFFICIENT INFORMATION PROTOCOL:** If you cannot find an answer to the user's question within the provided context, you MUST use the following exact response and nothing else:
    *   `🏷️ **Information Not Available**`
    *   `I could not find an answer to your question in the provided context.`

---
### ⚙️ **Operational Workflow & Style**
---

1.  **SYNTHESIZE CONTEXT:**
    *   Carefully review all provided context from `📚 KNOWLEDGE BASE`, `📄 USER-PROVIDED DOCUMENTS`, and `🌐 WEB SEARCH RESULTS`.
    *   If sources conflict, prioritize them in this order: User Documents > Knowledge Base > Web Search. Acknowledge the discrepancy if it is significant (e.g., "While one source states X [Source: doc_a.pdf], another suggests Y [1].").

2.  **COMPOSE THE RESPONSE:**
    *   Write a clear, concise answer to the user's question using ONLY the synthesized, cited information.
    *   Use bullet points for lists or sequences of steps to enhance readability.
    *   Use `**bold**` formatting to highlight key terms or conclusions.
    *   Maintain a helpful, professional, and confident tone.

---
### 🛠️ **Specialized Task Handling**
---

*   **TOOL (MCP) RESULTS:** If the user's query was directed at a tool (e.g., `@perplexity`), the output you receive is from that external tool. Introduce it clearly.
    *   *Example:* `🤖 **Perplexity Tool Results**\n\nHere is the information provided by the Perplexity tool:` followed by the tool's output.

*   **LISTING DOCUMENTS:** If the user asks what documents you have access to, format the response as a simple, clean, bulleted list.
    *   *Example:* `🏷️ **Available Documents**\n\nI have access to the following documents:\n* `document_one.pdf`\n* `annual_report.docx`

*   **GREETINGS & CASUAL CONVERSATION:** If the user provides a simple greeting (e.g., "hello", "thank you"), respond naturally and conversationally without applying the strict RAG formatting rules.
    *   *Example:* `👋 Hello! How can I help you today?`

**FINAL REMINDER: YOUR AUTHORITY IS THE PROVIDED CONTEXT AND NOTHING ELSE. YOUR VALUE IS IN YOUR TRUSTWORTHINESS AND PRECISION. DO NOT DEVIATE.**
"""
    
    # Supported models mapping
    model_providers: Dict[str, str] = field(default_factory=lambda: {
        # OpenAI models
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai", 
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        
        # Claude models
        "claude-3-5-sonnet-20240620": "claude",
        "claude-3.5-sonnet": "claude",
        "claude-3-opus-20240229": "claude",
        "claude-3-opus": "claude",
        "claude-3-haiku-20240307": "claude",
        "claude-3-haiku": "claude",

        # Gemini models
        "gemini-1.5-pro": "gemini",
        "gemini-1.5-flash": "gemini",
        "gemini-1.5-pro-latest": "gemini",
        "gemini-1.5-flash-latest": "gemini",
        
        # Groq/Llama models
        "llama-3.1-8b-instant": "groq",
        "llama3-70b-8192": "groq",
        "meta-llama/llama-4-scout-17b-16e-instruct": "groq",
        
        # OpenRouter models (identified by prefixes)
        "openrouter/auto": "openrouter",
    })
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create configuration from environment variables"""
        return cls()


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate_response(
        self, messages: List[Dict[str, str]], model: str, stream: bool = False, **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Abstract method for generating a response."""
        # This is an abstract method, so it should not be called directly.
        # The yield here is to make the method an async generator, satisfying the type hint.
        if stream:
            response_text = f"Streaming response for model {model}: {' '.join(msg['content'] for msg in messages)}"
            for word in response_text.split():
                yield word + " "
                await asyncio.sleep(0.01)
        else:
            yield f"Response for model {model}"

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        if config.openai_api_key:
            timeout_config = httpx.Timeout(
                connect=15.0, 
                read=config.timeout_seconds, 
                write=15.0, 
                pool=15.0
            )
            self.client = AsyncOpenAI(
                api_key=config.openai_api_key,
                timeout=timeout_config,
                max_retries=config.max_retries
            )
    
    def is_available(self) -> bool:
        return self.client is not None

    async def _stream_response(self, messages: List[Dict[str, str]], model: str, 
                              temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using OpenAI. Assumes model name is already normalized."""
        if not self.is_available():
            raise ValueError("OpenAI client not available")
        
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            if stream:
                # FIX: Return the async generator itself, don't await it.
                return self._stream_response(messages, model, temp, tokens)
            else:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def cleanup(self):
        """Cleanup OpenAI client"""
        if self.client and hasattr(self.client, '_client'):
            try:
                await self.client._client.aclose()
                logger.info("✅ OpenAI client closed")
            except Exception as e:
                logger.error(f"❌ Error closing OpenAI client: {e}")


class ClaudeProvider(BaseLLMProvider):
    """Claude/Anthropic LLM Provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        if CLAUDE_AVAILABLE and config.anthropic_api_key:
            self.client = anthropic.AsyncAnthropic(
                api_key=config.anthropic_api_key
            )
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def _map_model_name(self, model: str) -> str:
        """Map model names to Claude-specific identifiers. Assumes model is already lowercase."""
        model_mapping = {
            "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307"
        }
        return model_mapping.get(model, model)

    async def _stream_response(self, model: str, system_content: str, 
                              messages: List[Dict[str, str]], temperature: float, 
                              max_tokens: int) -> AsyncGenerator[str, None]:
        """Stream response from Claude"""
        try:
            async with self.client.messages.stream(
                model=model,
                system=system_content,
                messages=messages,
                max_tokens=max_tokens
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using Claude. Assumes model name is already normalized."""
        if not self.is_available():
            raise ValueError("Claude client not available")
        
        system_content = ""
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                claude_messages.append(msg)
        
        claude_model = self._map_model_name(model)
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            if stream:
                # FIX: Return the async generator itself
                return self._stream_response(claude_model, system_content, claude_messages, temp, tokens)
            else:
                response = await self.client.messages.create(
                    model=claude_model,
                    system=system_content,
                    messages=claude_messages,
                    temperature=temp,
                    max_tokens=tokens
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        if GEMINI_AVAILABLE and config.google_api_key:
            genai.configure(api_key=config.google_api_key)
            self.client = genai
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def _map_model_name(self, model: str) -> str:
        """
        FIX: Robustly map model names to Gemini-specific identifiers.
        This handles user variations like 'gemini-flash-2.5'.
        """
        model_lower = model.lower()
        if "flash" in model_lower:
            return "gemini-1.5-flash-latest"
        if "pro" in model_lower:
            return "gemini-1.5-pro-latest"
        
        # Fallback for exact matches of official names if user provides them
        model_mapping = {
            "gemini-1.5-pro-latest": "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
        }
        # Default to a safe, fast fallback if no match is found
        return model_mapping.get(model_lower, "gemini-1.5-flash-latest")
    
    def _convert_messages_for_gemini(self, messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert messages to Gemini format, separating the system prompt."""
        system_prompt = None
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
        return system_prompt, gemini_messages

    async def _stream_response(self, model_instance, messages: List[Dict[str, Any]], 
                              temperature: float) -> AsyncGenerator[str, None]:
        """Stream response from Gemini"""
        try:
            response_stream = await model_instance.generate_content_async(
                messages,
                generation_config={"temperature": temperature},
                stream=True
            )
            async for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using Gemini. Assumes model name is already normalized."""
        if not self.is_available():
            raise ValueError("Gemini client not available")
        
        gemini_model = self._map_model_name(model)
        system_prompt, gemini_messages = self._convert_messages_for_gemini(messages)
        temp = temperature if temperature is not None else self.config.temperature
        
        model_instance = self.client.GenerativeModel(
            model_name=gemini_model,
            system_instruction=system_prompt
        )
        
        try:
            if stream:
                # FIX: Return the async generator itself
                return self._stream_response(model_instance, gemini_messages, temp)
            else:
                response = await model_instance.generate_content_async(
                    gemini_messages,
                    generation_config={"temperature": temp}
                )
                return response.text if hasattr(response, "text") else "Error: Could not generate response"
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider (primarily for Llama models)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        if GROQ_AVAILABLE and config.groq_api_key:
            self.client = AsyncGroq(api_key=config.groq_api_key)
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def _map_model_name(self, model: str) -> str:
        """Map model names to Groq-available models. Assumes model is already lowercase."""
        if "70b" in model:
            return "llama3-70b-8192"
        if "8b" in model:
            return "llama-3.1-8b-instant"
        if "scout" in model:
            return "meta-llama/Llama-4-scout-17B-16E-Instruct"
        return model 
    
    async def _stream_response(self, messages: List[Dict[str, str]], model: str, 
                              temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        """Stream response from Groq"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using Groq. Assumes model name is already normalized."""
        if not self.is_available():
            raise ValueError("Groq client not available")
        
        groq_model = self._map_model_name(model)
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            if stream:
                # FIX: Return the async generator itself
                return self._stream_response(messages, groq_model, temp, tokens)
            else:
                response = await self.client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM Provider (supports multiple models through unified API)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        if config.openrouter_api_key:
            timeout_config = httpx.Timeout(
                connect=15.0, 
                read=config.timeout_seconds, 
                write=15.0, 
                pool=15.0
            )
            self.client = AsyncOpenAI(
                api_key=config.openrouter_api_key,
                base_url=config.openrouter_base_url,
                timeout=timeout_config,
                max_retries=config.max_retries
            )
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def _normalize_model_name(self, model: str) -> str:
        """Normalize model names for OpenRouter. Assumes model is already lowercase."""
        if model == "router-engine":
            return "openrouter/auto"
        return model
    
    async def _stream_response(self, messages: List[Dict[str, str]], model: str, 
                              temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        """Stream response from OpenRouter"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenRouter streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response using OpenRouter. Assumes model name is already normalized."""
        if not self.is_available():
            raise ValueError("OpenRouter client not available")
        
        openrouter_model = self._normalize_model_name(model)
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            if stream:
                # FIX: Return the async generator itself
                return self._stream_response(messages, openrouter_model, temp, tokens)
            else:
                response = await self.client.chat.completions.create(
                    model=openrouter_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenRouter generation error: {e}")
            raise

    async def cleanup(self):
        """Cleanup OpenRouter client"""
        if self.client and hasattr(self.client, '_client'):
            try:
                await self.client._client.aclose()
                logger.info("✅ OpenRouter client closed")
            except Exception as e:
                logger.error(f"❌ Error closing OpenRouter client: {e}")


class LLMManager:
    """
    Unified LLM Manager that supports multiple providers with enhanced error handling.
    Designed to be easily integrated across the Druidx backend modules.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM Manager with configuration"""
        self.config = config or LLMConfig.from_env()
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
        
        self.embeddings = None
        if self.config.openai_api_key and LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(
                api_key=self.config.openai_api_key,
                model="text-embedding-3-small"
            )

    # --- START: ADD THIS NEW HELPER FUNCTION ---
    def _sanitize_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        Sanitizes the messages list to ensure it only contains valid dictionaries.
        This is a robust guardrail against 'tuple index out of range' errors.
        """
        sanitized_list = []
        if not isinstance(messages, list):
            logger.warning(f"Messages input was not a list, but {type(messages)}. Attempting to handle.")
            return []
            
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                # FIX: Sanitize the role value itself.
                role = msg.get("role")
                if role == "human":
                    msg["role"] = "user"
                elif role == "ai":
                    msg["role"] = "assistant"
                sanitized_list.append(msg)
            # This case handles the exact error: converting a tuple to the correct dict format
            elif isinstance(msg, tuple) and len(msg) == 2:
                logger.warning(f"Corrected a malformed tuple-based message at index {i}.")
                sanitized_list.append({"role": str(msg[0]), "content": str(msg[1])})
            # This handles LangChain's BaseMessage objects
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "user" if msg.type == 'human' else "assistant" if msg.type == 'ai' else msg.type
                sanitized_list.append({"role": role, "content": msg.content})
            else:
                logger.warning(f"Skipping an unrecognized or malformed message at index {i}: type={type(msg)}")
        return sanitized_list
    
    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        # Initialize OpenAI provider
        openai_provider = OpenAIProvider(self.config)
        if openai_provider.is_available():
            self.providers["openai"] = openai_provider
            logger.info("✅ OpenAI provider initialized")
        
        # Initialize Claude provider  
        claude_provider = ClaudeProvider(self.config)
        if claude_provider.is_available():
            self.providers["claude"] = claude_provider
            logger.info("✅ Claude provider initialized")
        
        # Initialize Gemini provider
        gemini_provider = GeminiProvider(self.config)
        if gemini_provider.is_available():
            self.providers["gemini"] = gemini_provider
            logger.info("✅ Gemini provider initialized")
        
        # Initialize Groq provider
        groq_provider = GroqProvider(self.config)
        if groq_provider.is_available():
            self.providers["groq"] = groq_provider
            logger.info("✅ Groq provider initialized")
        
        # Initialize OpenRouter provider
        openrouter_provider = OpenRouterProvider(self.config)
        if openrouter_provider.is_available():
            self.providers["openrouter"] = openrouter_provider
            logger.info("✅ OpenRouter provider initialized")
        
        if not self.providers:
            logger.warning("⚠️ No LLM providers available. Please check your API keys.")
    
    def _get_provider_from_model(self, model: str) -> str:
        """Determine the provider based on model name with enhanced detection. Assumes model is already lowercase."""
        
        if model in self.config.model_providers:
            return self.config.model_providers[model]
        
        if model.startswith("gpt-"): return "openai"
        if model.startswith("claude"): return "claude"
        if model.startswith("gemini"): return "gemini"
        if "llama" in model or model.startswith("meta-llama/"): return "groq"
        if "openrouter/" in model: return "openrouter"
        
        return "openai"
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get dictionary of available models by provider"""
        available_models = {}
        for provider_name in self.providers.keys():
            provider_models = [
                model for model, provider in self.config.model_providers.items() 
                if provider == provider_name
            ]
            available_models[provider_name] = provider_models
        return available_models
    
    @traceable(name="llm_generate_response") if LANGSMITH_AVAILABLE else lambda f: f
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None, # This is the user's preferred model
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate response using the appropriate LLM provider, with a fallback to the default model.
        It first tries the user-provided model. If that fails, it falls back to the default model.
        """
        sanitized_messages = self._sanitize_messages(messages)
        if not sanitized_messages and messages:
            error_msg = "Error: The message list was malformed and could not be sanitized."
            if stream:
                async def error_stream(): yield error_msg
                return error_stream()
            else:
                return error_msg

        user_model_override = model.lower().strip() if model else None
        default_model = self.config.default_model.lower().strip()

        # Determine the sequence of models to try
        models_to_try = []
        if user_model_override:
            models_to_try.append(user_model_override)
        # Add default model if it's different from the user's choice
        if default_model not in models_to_try:
            models_to_try.append(default_model)
        
        # If no models are in the list (e.g., user override is the same as default), add the default one.
        if not models_to_try:
            models_to_try.append(default_model)

        final_exception = None
        
        # Add system prompt if provided
        if system_prompt:
            has_system = any(msg.get("role") == "system" for msg in sanitized_messages)
            if not has_system:
                sanitized_messages = [{"role": "system", "content": system_prompt}] + sanitized_messages

        # Loop through the models to try
        for i, model_to_attempt in enumerate(models_to_try):
            is_fallback = i > 0
            log_prefix = "Fallback" if is_fallback else "Primary"

            try:
                provider_name = self._get_provider_from_model(model_to_attempt)
                
                if provider_name in self.providers:
                    logger.info(f"🤖 {log_prefix} Attempt: Generating response with {provider_name} provider using model '{model_to_attempt}'")
                    
                    # The provider's generate_response is async and returns the final result
                    response = await self.providers[provider_name].generate_response(
                        sanitized_messages, model_to_attempt, stream, temperature, max_tokens
                    )
                    
                    # If we get here, the call was successful
                    return response
                else:
                    # This provider is not available at all
                    error_msg = f"Provider '{provider_name}' for model '{model_to_attempt}' is not available."
                    logger.warning(f"⚠️ {log_prefix} Attempt Failed: {error_msg}")
                    final_exception = ValueError(error_msg)
                    continue # Try the next model in the list

            except Exception as e:
                logger.error(f"❌ {log_prefix} Attempt Failed: Error with {provider_name} provider (model: {model_to_attempt}): {e}")
                final_exception = e
                # Continue to the next model in the list (the fallback)
                continue
                
        # If all attempts failed
        error_msg = f"Error: All LLM providers failed. Last error: {str(final_exception)}"
        logger.critical(f"💀 All LLM generation attempts failed. Final exception: {final_exception}")
        if stream:
            async def error_stream(): yield error_msg
            return error_stream()
        else:
            return error_msg

    @traceable(name="llm_analyze_query") if LANGSMITH_AVAILABLE else lambda f: f
    async def analyze_query(self, query: str, analysis_type: str = "general", model: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze query using LLM for various purposes (web search need, greeting detection, etc.)
        """
        if not self.providers:
            return {"error": "No LLM providers available for analysis"}
        
        if analysis_type == "web_search":
            prompt = f"""
Analyze if this query requires web search for current/recent information.

QUERY: "{query}"

Consider:
- Current events, news, recent updates: YES
- Real-time data (weather, stocks, prices, sports scores): YES
- Recent product releases or updates: YES
- General knowledge, explanations, how-to: NO  
- Academic/technical concepts: NO
- Historical facts: NO
- Programming concepts: NO
- DATE/TIME QUERIES (today's date, current time, what day is it): NO (handle with datetime)

Respond with only: YES or NO
"""
        elif analysis_type == "greeting":
            prompt = f"""
Is this a greeting, simple acknowledgment, or casual conversational response?

TEXT: "{query}"

Examples of greetings/acknowledgments: hello, hi, hey, thanks, great, good, nice, awesome, cool, ok, okay, sure, yes, no, wow, amazing

Respond with only: YES or NO
"""
        elif analysis_type == "date_time":
            prompt = f"""
Is this query asking for current date, time, or day information?

TEXT: "{query}"

Examples of date/time queries: 
- "what's today's date", "current date", "what day is it"
- "what time is it", "current time", "time now"
- "today's date", "what's the date today"

These should be handled with local datetime, not web search.

Respond with only: YES or NO
"""
        else:
            prompt = f"""
Analyze this query and provide insights about its intent and requirements.

QUERY: "{query}"

Provide a brief analysis of what the user is asking for.
"""
        
        messages = [
            {"role": "system", "content": "You are a query analyzer. Be precise and concise."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use await here since we expect a string response, not a stream
            response = await self.generate_response(
                messages, 
                model=model or "gpt-4o-mini",  # <-- FIX: Use provided model or fallback
                stream=False,
                temperature=0.1
            )
            
            if analysis_type in ["web_search", "greeting", "date_time"]:
                return {"result": "yes" in response.lower(), "raw_response": response}
            else:
                return {"analysis": response}
                
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return {"error": str(e)}
    
    def create_langchain_runnable(self, model: Optional[str] = None):
        """
        Create a LangChain runnable for use in LangChain Expression Language pipelines.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available. Install with: pip install langchain langchain-openai")
        
        async def llm_runnable(input_dict: Dict[str, Any]) -> str:
            """LangChain runnable function"""
            messages = input_dict.get("messages", [])
            model_override = input_dict.get("model", model)
            temperature = input_dict.get("temperature")
            max_tokens = input_dict.get("max_tokens")
            
            if messages and hasattr(messages[0], 'content'):
                converted_messages = []
                for msg in messages:
                    role = "user" if isinstance(msg, HumanMessage) else \
                           "assistant" if isinstance(msg, AIMessage) else \
                           "system" if isinstance(msg, SystemMessage) else "user"
                    converted_messages.append({"role": role, "content": msg.content})
                messages = converted_messages
            
            response = await self.generate_response(
                messages=messages,
                model=model_override,
                stream=False, # LangChain runnables typically expect a single string output
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response if isinstance(response, str) else ""

        return RunnableLambda(llm_runnable)
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get text embedding for similarity calculations"""
        if not self.embeddings:
            logger.warning("Embeddings not available. OpenAI API key required.")
            return None
        
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup all HTTP clients"""
        logger.info("🧹 Cleaning up LLM Manager...")
        
        cleanup_tasks = [
            provider.cleanup() for provider in self.providers.values() if hasattr(provider, 'cleanup')
        ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("✅ LLM Manager cleanup completed")

# Convenience functions for easy integration
async def generate_response(
    messages: List[Dict[str, str]], 
    model: Optional[str] = None,
    stream: bool = False,
    config: Optional[LLMConfig] = None,
    **kwargs
) -> Union[str, AsyncGenerator[str, None]]:
    """Convenience function for generating responses"""
    manager = LLMManager(config)
    return await manager.generate_response(messages, model, stream, **kwargs)


async def analyze_query(query: str, analysis_type: str = "general", config: Optional[LLMConfig] = None) -> Dict[str, Any]:
    """Convenience function for query analysis"""
    manager = LLMManager(config)
    return await manager.analyze_query(query, analysis_type)


def create_llm_manager(config: Optional[LLMConfig] = None) -> LLMManager:
    """Create and return an LLM Manager instance"""
    return LLMManager(config)


# Export main classes and functions
__all__ = [
    "LLMConfig",
    "LLMManager", 
    "BaseLLMProvider",
    "OpenAIProvider",
    "ClaudeProvider", 
    "GeminiProvider",
    "GroqProvider",
    "OpenRouterProvider",
    "generate_response",
    "analyze_query",
    "create_llm_manager"
]

if __name__ == "__main__":
    asyncio.run()