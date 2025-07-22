import os
import asyncio
import logging
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Tuple
from urllib.parse import urlparse
from llm_model import LLMManager
from qdrant_client import QdrantClient

# Core imports
import httpx

# LangChain imports (following user requirements)
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_message_histories import ChatMessageHistory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.error("LangChain packages not found. Please install: pip install langchain langchain-community langchain-openai")

# LangSmith imports (following user requirements)
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith not available. Install with: pip install langsmith")

# Import storage system (CRITICAL INTEGRATION)
try:
    from storage import CloudflareR2Storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logging.error("storage.py not found. Please ensure it's in the same directory.")

# Import all integrated modules
try:
    from llm_model import LLMManager, LLMConfig
    LLM_MODEL_AVAILABLE = True
except ImportError:
    LLM_MODEL_AVAILABLE = False
    logging.error("llm_model.py not found. Please ensure it's in the same directory.")

try:
    from mcp_core_file import MCPCore, MCPConfiguration
    MCP_CORE_AVAILABLE = True
except ImportError:
    MCP_CORE_AVAILABLE = False
    logging.error("mcp_core_file.py not found. Please ensure it's in the same directory.")

try:
    from rag_code import RAGPipeline, RAGConfig, DocumentProcessor, VectorStoreManager
    RAG_CODE_AVAILABLE = True
except ImportError:
    RAG_CODE_AVAILABLE = False
    logging.error("rag_code.py not found. Please ensure it's in the same directory.")

try:
    from websearch_code import WebSearch, WebSearchConfig
    WEBSEARCH_AVAILABLE = True
except ImportError:
    WEBSEARCH_AVAILABLE = False
    logging.error("websearch_code.py not found. Please ensure it's in the same directory.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatbotConfig:
    """Unified configuration for the chatbot agent with R2 storage integration"""
    
    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    
    # CloudflareR2 Storage Keys (NEW - CRITICAL INTEGRATION)
    cloudflare_account_id: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    cloudflare_access_key_id: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_ACCESS_KEY_ID", ""))
    cloudflare_secret_access_key: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY", ""))
    cloudflare_bucket_name: str = field(default_factory=lambda: os.getenv("CLOUDFLARE_BUCKET_NAME", "rag-documents"))
    
    # Vector Database
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    # Add a field to hold the collection name, which can now be overridden.
    qdrant_collection_name: str = "default_knowledge_base"
    
    # LLM Settings
    default_model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 2000
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    use_hybrid_search: bool = True
    
    # Storage Settings (NEW)
    enable_r2_storage: bool = True
    fallback_to_local: bool = True
    temp_processing_path: str = "local_rag_data/temp_downloads"
    file_deletion_hours: int = 72  # Auto-delete files after 72 hours
    
    # Session Management
    memory_expiry_minutes: int = 10
    max_conversation_turns: int = 10
    max_context_messages: int = 6
    
    # Web Search Settings
    web_search_max_results: int = 5
    
    # System Prompts - FIXED: Strict context-only responses
    default_system_prompt: str = """You are Insight Agent, a precision AI assistant. Your primary mission is to provide accurate, clear, and trustworthy answers based **exclusively** on the information provided to you in the 'CONTEXT' section.

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
    
    @classmethod
    def from_env(cls) -> 'ChatbotConfig':
        """Create configuration from environment variables"""
        return cls()
    
    def validate_storage_config(self) -> bool:
        """Validate storage configuration"""
        if not self.enable_r2_storage:
            return True
        
        required_keys = [
            self.cloudflare_account_id,
            self.cloudflare_access_key_id, 
            self.cloudflare_secret_access_key
        ]
        
        return all(key.strip() for key in required_keys)
    
    def to_llm_config(self) -> 'LLMConfig':
        """Convert to LLM configuration"""
        if not LLM_MODEL_AVAILABLE:
            raise ImportError("llm_model module not available")
        
        return LLMConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            langsmith_api_key=self.langsmith_api_key,
            default_model=self.default_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            default_system_prompt=self.default_system_prompt
        )
    
    def to_rag_config(self, storage_client: Optional['CloudflareR2Storage'] = None) -> 'RAGConfig':
        """Convert to RAG configuration with storage integration"""
        if not RAG_CODE_AVAILABLE:
            raise ImportError("rag_code module not available")
        
        config = RAGConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            langsmith_api_key=self.langsmith_api_key,
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            qdrant_collection_name=self.qdrant_collection_name,
            llm_model=self.default_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            retrieval_k=self.retrieval_k,
            use_hybrid_search=self.use_hybrid_search,
            system_prompt=self.default_system_prompt
        )
        
        # Add storage integration (CRITICAL)
        if storage_client:
            config.storage_client = storage_client
        
        return config
    
    def to_websearch_config(self, storage_client: Optional['CloudflareR2Storage'] = None) -> 'WebSearchConfig':
        """Convert to web search configuration with storage integration"""
        if not WEBSEARCH_AVAILABLE:
            raise ImportError("websearch_code module not available")
        
        config = WebSearchConfig(
            tavily_api_key=self.tavily_api_key,
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            langsmith_api_key=self.langsmith_api_key,
            max_results=self.web_search_max_results,
            analysis_model=self.default_model,
            analysis_temperature=self.temperature,
        )
        
        # Add storage integration
        if storage_client:
            config.storage_client = storage_client
        
        return config
    
    def to_mcp_config(self, storage_client: Optional['CloudflareR2Storage'] = None) -> 'MCPConfiguration':
        """Convert to MCP configuration with storage integration"""
        if not MCP_CORE_AVAILABLE:
            raise ImportError("mcp_core_file module not available")
        
        config = MCPConfiguration(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            analysis_model=self.default_model,
            analysis_temperature=self.temperature
        )
        
        # Add storage integration
        if storage_client:
            config.storage_client = storage_client
        
        return config


class ChatbotAgent:
    """
    Enhanced Chatbot Agent integrating RAG, Web Search, MCP capabilities, and R2 Storage
    Uses LangChain Expression Language for RAG pipeline construction
    Now includes proper CloudflareR2Storage integration matching rag.py pattern
    """
    
    def __init__(self, 
                config: Optional[ChatbotConfig] = None, 
                storage_client: Optional['CloudflareR2Storage'] = None,
                llm_manager: Optional[LLMManager] = None,
                qdrant_client: Optional[QdrantClient] = None):         
        """Initialize the chatbot agent with all integrated modules including storage"""
        self.config = config or ChatbotConfig.from_env()
        
        # Session management
        self.sessions: Dict[str, ChatMessageHistory] = {}
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        
        self.llm_manager = llm_manager
        self.qdrant_client = qdrant_client


        # Initialize storage system FIRST (CRITICAL)
        self.storage_client = storage_client
        self._initialize_storage() # This will use the provided client if available
        
        # Initialize all modules with storage integration
        self._initialize_modules()
        
        self.has_vision_capability = self.config.default_model.lower() in [
            "gpt-4o", "gpt-4o-mini", "gpt-4-vision", "gpt-4-turbo",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17", "gemini-2.0-flash",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3.5-sonnet-20240620",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ] or "vision" in self.config.default_model.lower()

        if self.has_vision_capability:
            logger.info(f"✅ Agent configured with a vision-capable model: {self.config.default_model}")
        else:
            logger.warning(f"⚠️ Agent's model may not support vision: {self.config.default_model}. Image processing will be handled by RAG pipeline if possible.")

        
        # Setup LangSmith if available
        if LANGSMITH_AVAILABLE and self.config.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.config.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"
            LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
            logger.info("✅ LangSmith tracing enabled")
    
    def _initialize_storage(self):
        """Initialize CloudflareR2Storage system if not already provided."""
        # If a storage client was already passed in, do nothing.
        if self.storage_client:
            logger.info("✅ ChatbotAgent using pre-initialized storage client.")
            return

        # Otherwise, create a new one.
        try:
            if not STORAGE_AVAILABLE:
                logger.error("❌ Storage module not available - storage.py not found")
                self.storage_client = None
                return
            
            if not self.config.enable_r2_storage:
                logger.info("🗄️ R2 storage disabled by configuration")
                self.storage_client = None
                return
            
            self.storage_client = CloudflareR2Storage()
            
            if self.storage_client.use_local_fallback:
                logger.warning("⚠️ R2 storage initialized with local fallback")
            else:
                logger.info("✅ CloudflareR2Storage initialized successfully by ChatbotAgent")
                
        except Exception as e:
            logger.error(f"❌ Error initializing storage within ChatbotAgent: {e}")
            self.storage_client = None
    
    def _initialize_modules(self):
        """Initialize all integrated modules with proper error handling and storage integration"""
        try:
            if not self.llm_manager:
                logger.warning("No global LLM Manager provided, creating a local instance.")
                if LLM_MODEL_AVAILABLE:
                    llm_config = self.config.to_llm_config()
                    self.llm_manager = LLMManager(llm_config)
                else:
                    self.llm_manager = None
                    logger.error("❌ LLM Manager not available")
            else:
                logger.info("✅ Agent is using the global LLM Manager.")
            
            # Initialize RAG Pipeline with storage integration (CRITICAL)
            if RAG_CODE_AVAILABLE:
                # Create the RAG configuration object from the main ChatbotConfig
                rag_config = self.config.to_rag_config(self.storage_client)

                self.rag_pipeline = RAGPipeline(
                   config=rag_config, 
                   r2_storage_client=self.storage_client,
                   llm_manager=self.llm_manager,
                   qdrant_client=self.qdrant_client
                   )
                logger.info("✅ RAG Pipeline initialized with storage integration")
            else:
                self.rag_pipeline = None
                logger.error("❌ RAG Pipeline not available")
            
            # Initialize Web Search with storage integration
            if WEBSEARCH_AVAILABLE:
                websearch_config = self.config.to_websearch_config(self.storage_client)
                self.web_search = WebSearch(config=websearch_config, llm_manager=self.llm_manager)
                logger.info("✅ Web Search initialized with storage integration")
            else:
                self.web_search = None
                logger.error("❌ Web Search not available")
            
            # Initialize MCP Core with storage integration
            if MCP_CORE_AVAILABLE:
                mcp_config = self.config.to_mcp_config(self.storage_client)
                self.mcp_core = MCPCore(config=mcp_config, llm_manager=self.llm_manager)
                logger.info("✅ MCP Core initialized with storage integration")
            else:
                self.mcp_core = None
                logger.error("❌ MCP Core not available")
                
        except Exception as e:
            logger.error(f"❌ Error initializing modules: {e}")
            raise
    
    async def initialize(self):
        """Async initialization of modules that require it"""
        try:
            if self.rag_pipeline:
                await self.rag_pipeline.initialize()
                logger.info("✅ RAG Pipeline async initialization completed")
                
            # Initialize storage async operations if needed
            if self.storage_client and hasattr(self.storage_client, 'initialize_async'):
                await self.storage_client.initialize_async()
                logger.info("✅ Storage client async initialization completed")
                
        except Exception as e:
            logger.error(f"❌ Error in async initialization: {e}")
            raise
    
    def _get_session_memory(self, session_id: str) -> ChatMessageHistory:
        """Get or create session memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
            self.session_contexts[session_id] = {
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0
            }
        
        # Update last activity
        self.session_contexts[session_id]["last_activity"] = datetime.now()
        return self.sessions[session_id]
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, context in self.session_contexts.items():
            last_activity = context.get("last_activity", current_time)
            minutes_since_activity = (current_time - last_activity).total_seconds() / 60
            
            if minutes_since_activity > self.config.memory_expiry_minutes:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.sessions.pop(session_id, None)
            self.session_contexts.pop(session_id, None)
            logger.info(f"🧹 Cleaned up expired session: {session_id}")
    
    def _format_chat_history(self, memory: ChatMessageHistory) -> List[Dict[str, str]]:
        """Format chat history for LLM consumption"""
        messages = []
        for message in memory.messages[-self.config.max_context_messages:]:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                if message.type == 'human':
                    messages.append({"role": "user", "content": message.content})
                elif message.type == 'ai':
                    messages.append({"role": "assistant", "content": message.content})
        return messages
    
    @traceable(name="chatbot_analyze_query") if LANGSMITH_AVAILABLE else lambda f: f
    async def analyze_query_intent(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Enhanced query analysis with new priority system...
        """
        try:
            analysis_results = {}

            # Add a check for document listing queries.
            doc_listing_phrases = [
                "what document do you have", "what documents do you have", 
                "list your documents", "show me your documents", "what is in your knowledge base"
            ]
            query_lower = query.lower().strip().rstrip('?')
            analysis_results["is_document_listing_query"] = any(phrase in query_lower for phrase in doc_listing_phrases)
            
            # PRIORITY 1: Check for explicit MCP server references (@{servername})
            mcp_detection = self._detect_mcp_reference(query)
            analysis_results.update(mcp_detection)
            
            # PRIORITY 2: Fast greeting detection for immediate response
            if self.llm_manager:
                greeting_analysis = await self.llm_manager.analyze_query(query, "greeting")
                analysis_results["is_greeting"] = greeting_analysis.get("result", False)
            else:
                # Enhanced fallback with more greeting patterns
                greeting_words = ["hello", "hi", "hey", "thanks", "thank you", "great", "good", "nice", 
                                "awesome", "cool", "ok", "okay", "sure", "yes", "no", "wow"]
                analysis_results["is_greeting"] = any(word in query.lower().split() for word in greeting_words)
            
            # PRIORITY 3: Analyze web search necessity (for combining with RAG)
            if self.web_search and self.web_search.is_available():
                should_search = await self.web_search.should_search(query, chat_history)
                analysis_results["web_search_needed"] = should_search
            else:
                analysis_results["web_search_needed"] = False
            
            # Detect follow-up context
            analysis_results["is_follow_up"] = self._detect_follow_up(query, chat_history)
            
            # Extract URLs if any
            if self.web_search and hasattr(self.web_search, 'client'):
                urls = self.web_search.client.extract_urls_from_query(query)
                analysis_results["detected_urls"] = urls
            else:
                analysis_results["detected_urls"] = []
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error in query intent analysis: {e}")
            return {
                "mcp_enabled": False,
                "mcp_server": None,
                "cleaned_query": query,
                "web_search_needed": False,
                "is_greeting": False,
                "is_follow_up": False,
                "detected_urls": [],
                "error": str(e)
            }

    def _detect_mcp_reference(self, query: str) -> Dict[str, Any]:
        """
        Enhanced MCP server reference detection using improved regex patterns and fuzzy matching
        Returns: {mcp_enabled: bool, mcp_server: str|None, cleaned_query: str}
        """
        import re
        
        # Enhanced MCP detection patterns (following rag.py pattern)
        mcp_patterns = [
            r'@([a-zA-Z0-9\-_]+)(?:\s|$|:)',  # @servername with word boundary - HIGHEST PRIORITY
            r'@\{([^}]+)\}',                    # @{servername}
            r'@([a-zA-Z0-9\-_]+):',            # @servername:
            r'@mcp(?:\s|$)'                    # Generic @mcp reference
        ]
        
        for pattern in mcp_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                # Get the first match
                if pattern == r'@mcp(?:\s|$)':
                    # Generic MCP reference
                    cleaned_query = re.sub(r'@mcp:?\s*', '', query, flags=re.IGNORECASE).strip()
                    return {
                        "mcp_enabled": True,
                        "mcp_server": None,  # Let system choose based on schema
                        "cleaned_query": cleaned_query or query
                    }
                else:
                    # Specific server reference - use the first match
                    server_name = matches[0] if isinstance(matches[0], str) else matches[0]
                    server_name = server_name.strip().lower()
                    
                    # Clean the query by removing the @server reference
                    # More comprehensive cleaning pattern
                    cleaning_patterns = [
                        r'@\{?' + re.escape(server_name) + r'\}?(?::|\s|$)',  # @server or @{server}
                        r'@' + re.escape(server_name) + r'(?:\s|:|$)',         # @server
                    ]
                    
                    cleaned_query = query
                    for clean_pattern in cleaning_patterns:
                        cleaned_query = re.sub(clean_pattern, '', cleaned_query, flags=re.IGNORECASE).strip()
                    
                    logger.info(f"🎯 Enhanced: Detected MCP server reference: @{server_name}")
                    return {
                        "mcp_enabled": True,
                        "mcp_server": server_name,
                        "cleaned_query": cleaned_query or query
                    }
        
        return {
            "mcp_enabled": False,
            "mcp_server": None,
            "cleaned_query": query
        }
        
    def _detect_follow_up(self, query: str, chat_history: List[Dict[str, str]]) -> bool:
        """Simple follow-up detection"""
        if not chat_history:
            return False
        
        follow_up_indicators = [
            "what about", "how about", "tell me more", "can you explain", 
            "elaborate", "also", "furthermore", "additionally", "and"
        ]
        
        return any(indicator in query.lower() for indicator in follow_up_indicators)
    
    @traceable(name="chatbot_gather_context") if LANGSMITH_AVAILABLE else lambda f: f
    async def gather_context_documents(
        self, 
        query: str, 
        session_id: str,
        chat_history: List[Dict[str, str]],
        enable_web_search: bool = True,
        user_documents: Optional[List[str]] = None
    ) -> Dict[str, List[Document]]:
        """Gather context documents from all available sources"""
        context_docs = {
            "rag_docs": [],
            "web_docs": [],
            "user_docs": []
        }
        
        try:
            # Step 1: Gather RAG documents from the permanent knowledge base
            if self.rag_pipeline:
                try:
                    rag_docs = await self.rag_pipeline.get_relevant_documents(query, self.config.retrieval_k)
                    context_docs["rag_docs"] = rag_docs
                    logger.info(f"📚 Retrieved {len(rag_docs)} RAG documents")
                except Exception as e:
                    logger.error(f"❌ Error retrieving RAG documents: {e}")
            
            # Step 2: Gather web search documents if enabled
            if enable_web_search and self.web_search and self.web_search.is_available():
                try:
                    web_docs = await self.web_search.search(query, self.config.web_search_max_results)
                    context_docs["web_docs"] = web_docs
                    logger.info(f"🌐 Retrieved {len(web_docs)} web documents")
                except Exception as e:
                    logger.error(f"❌ Error retrieving web documents: {e}")
            
            # Step 3: Process user-provided documents for this specific query
            if user_documents and self.rag_pipeline and hasattr(self.rag_pipeline, 'document_processor'):
                try:
                    logger.info(f"👤 Processing {len(user_documents)} user-provided document URLs for this query...")
                    # Load documents directly from the provided URLs without indexing them
                    raw_docs = await self.rag_pipeline.document_processor.load_documents_from_urls(user_documents)
                    
                    if raw_docs:
                        split_docs = self.rag_pipeline.document_processor.split_documents(raw_docs)
                        # Add metadata to identify these as on-the-fly user docs
                        for doc in split_docs:
                            doc.metadata["source_type"] = "user_provided"
                        
                        context_docs["user_docs"] = split_docs
                        logger.info(f"👤 Added {len(split_docs)} chunks from user documents directly to the context.")
                    else:
                        logger.warning(f"👤 No content could be loaded from the provided user document URLs: {user_documents}")

                except Exception as e:
                    logger.error(f"❌ Error processing user document URLs: {e}", exc_info=True)
            
            return context_docs
            
        except Exception as e:
            logger.error(f"❌ Error gathering context documents: {e}")
            return context_docs
    
    def _create_rag_chain(self, documents: List[Document], query: str, chat_history: List[Dict[str, str]], system_prompt_override: Optional[str] = None) -> Any:
        """Create a LangChain Expression Language RAG chain"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available. Please install langchain packages.")
        
        def format_documents(docs: List[Document]) -> str:
            """
            Enhanced document formatting that creates inline citation markers and a final citation list.
            It strictly separates web URLs from knowledge base/user document sources.
            """
            if not docs:
                return "No relevant documents found."

            url_to_index_map: Dict[str, int] = {}
            web_citations: List[str] = []
            formatted_context_parts = []

            # --- Categorize and build citation list ---
            web_docs_content = []
            kb_docs_content = []
            user_docs_content = []

            for doc in docs:
                source_type = doc.metadata.get("source_type", "")
                # Check if it's a web search result by source_type or if the source is a non-file URL
                is_web = source_type == "web_search" or (
                    "http" in doc.metadata.get("source", "") and not doc.metadata.get("source", "").startswith("file://")
                )
                is_user = "user" in source_type or "user" in doc.metadata.get("source", "").lower()

                if is_web:
                    url = doc.metadata.get("url", doc.metadata.get("source")) # Use URL from metadata if available
                    if url and url not in url_to_index_map:
                        url_to_index_map[url] = len(web_citations) + 1
                        web_citations.append(url)
                    # Add a citation index only for web documents
                    doc.metadata["citation_index"] = url_to_index_map.get(url, 0)
                    web_docs_content.append(doc)
                elif is_user:
                    user_docs_content.append(doc)
                else:
                    kb_docs_content.append(doc)

            # --- Format sections for the LLM ---
            if web_docs_content:
                formatted_context_parts.append("## 🌐 **WEB SEARCH RESULTS**")
                for doc in web_docs_content:
                    title = doc.metadata.get('title', 'Web Result')
                    content = doc.page_content.strip()
                    # Inline citation marker like [1], [2], etc.
                    citation_marker = f" [{doc.metadata['citation_index']}]" if doc.metadata.get('citation_index') else ""
                    formatted_context_parts.append(f"### 📰 **{title}**\n**Content:** {content}{citation_marker}")

            if user_docs_content:
                formatted_context_parts.append("---" if formatted_context_parts else "")
                formatted_context_parts.append("## 📄 **USER-PROVIDED DOCUMENTS**")
                for doc in user_docs_content:
                    source_url = doc.metadata.get('source', 'Unknown User Document')
                    # Extract just the filename for the citation
                    doc_name = os.path.basename(urlparse(source_url).path)
                    content = doc.page_content.strip()
                    # Inline citation using the document name
                    citation_marker = f" [Source: {doc_name}]"
                    formatted_context_parts.append(f"### 📄 **Source:** {doc_name}\n**Content:** {content}{citation_marker}")

            if kb_docs_content:
                formatted_context_parts.append("---" if formatted_context_parts else "")
                formatted_context_parts.append("## 📚 **KNOWLEDGE BASE**")
                for doc in kb_docs_content:
                    source_url = doc.metadata.get('source', 'Unknown KB Document')
                    # Extract just the filename for the citation
                    doc_name = os.path.basename(urlparse(source_url).path)
                    content = doc.page_content.strip()
                    # Inline citation using the document name
                    citation_marker = f" [Source: {doc_name}]"
                    formatted_context_parts.append(f"### 📚 **Source:** {doc_name}\n**Content:** {content}{citation_marker}")

            # --- Append the URL citation list ONLY for web search results ---
            if web_citations:
                formatted_context_parts.append("---\n## 🔗 **CITATIONS**")
                for i, url in enumerate(web_citations, 1):
                    formatted_context_parts.append(f"[{i}] {url}")

            return "\n\n".join(formatted_context_parts)
        
        def get_current_time() -> str:
            """Get current timestamp"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def format_chat_history_summary() -> str:
            """Format chat history summary"""
            if not chat_history:
                return "No previous conversation."
            
            recent_history = chat_history[-3:]  # Last 3 exchanges
            formatted_history = []
            for msg in recent_history:
                role = "Human" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                formatted_history.append(f"{role}: {content}")
            
            return "\n".join(formatted_history)
        
        def get_chat_history_count() -> int:
            """Get chat history count"""
            return len(chat_history) if chat_history else 0
        
        # Create enhanced prompt template matching the new citation format
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_override or self.config.default_system_prompt),
            ("human", """📘 **CONTEXT:**
{context}

💬 **USER QUESTION:** {question}

CRITICAL INSTRUCTION: Your response MUST follow all rules in the system prompt. Synthesize an answer to the user's question using ONLY the provided context. Add citations for every piece of information as instructed: use `[Source: Document Name]` for documents and `[1]` for web results.""")
        ])

        # Create the RAG chain using LCEL
        rag_chain = (
            RunnableParallel({
                "context": RunnableLambda(lambda x: format_documents(documents)),
                "current_time": RunnableLambda(lambda x: get_current_time()),
                "question": RunnablePassthrough(),
                "chat_history_count": RunnableLambda(lambda x: get_chat_history_count()),
                "chat_history_summary": RunnableLambda(lambda x: format_chat_history_summary())
            })
            | prompt_template
            | self._get_llm_runnable()
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _format_chat_history_for_context(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for context inclusion"""
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for msg in chat_history[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def _get_llm_runnable(self, system_prompt_override: Optional[str] = None):
        """Get LLM as a runnable for LCEL chains"""
        if not self.llm_manager:
            raise ValueError("LLM Manager not available")
        
        async def process_llm_input(input_dict):
            """Process input through LLM Manager"""
            try:
                # Handle LangChain prompt format
                if hasattr(input_dict, 'to_string'):
                    formatted_prompt = input_dict.to_string()
                elif hasattr(input_dict, 'content'):
                    formatted_prompt = input_dict.content
                elif isinstance(input_dict, str):
                    formatted_prompt = input_dict
                elif isinstance(input_dict, dict):
                    formatted_prompt = input_dict.get("text", str(input_dict))
                else:
                    formatted_prompt = str(input_dict)
                
                # Convert to messages format
                messages = [{"role": "user", "content": formatted_prompt}]
                
                # Generate response
                response = await self.llm_manager.generate_response(
                    messages=messages,
                    model=self.config.default_model,
                    stream=False,
                    system_prompt=system_prompt_override
                )
                
                return response
            except Exception as e:
                logger.error(f"❌ Error in LLM runnable: {e}")
                return f"Error processing request: {str(e)}"
        
        return RunnableLambda(process_llm_input)
    
    @traceable(name="chatbot_query") if LANGSMITH_AVAILABLE else lambda f: f
    async def query(
        self, 
        session_id: str, 
        query: str,
        enable_web_search: bool = True,
        user_documents: Optional[List[str]] = None,
        model_override: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
        mcp_enabled: bool = False,
        mcp_schema: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Process a single query and return response"""
        try:
            # Clean up expired sessions
            self._cleanup_expired_sessions()
            
            # Get session memory
            memory = self._get_session_memory(session_id)
            chat_history = self._format_chat_history(memory)
            
            # Analyze query intent
            intent_analysis = await self.analyze_query_intent(query, chat_history)
            
            # Handle MCP requests if enabled
            if mcp_enabled and self.mcp_core and mcp_schema:
                try:
                    response_parts = []
                    async for response in self.mcp_core.execute_mcp_request(
                        query=query,
                        mcp_schema=mcp_schema,
                        chat_history=chat_history,
                        api_keys=api_keys,
                        detected_server_name=intent_analysis.get("mcp_server"),
                        model_override=model_override
                    ):
                        if response.get("type") == "content":
                            response_parts.append(response.get("data", ""))
                    
                    final_response = "".join(response_parts)
                    
                    # Save to memory
                    memory.add_user_message(query)
                    memory.add_ai_message(final_response)
                    
                    return {
                        "response": final_response,
                        "source": "mcp",
                        "session_id": session_id,
                        "intent_analysis": intent_analysis
                    }
                except Exception as e:
                    logger.error(f"❌ MCP request failed: {e}")
                    # Continue with regular processing
            
            # Determine if web search is needed
            should_web_search = enable_web_search and intent_analysis.get("web_search_needed", False)
            
            # Gather context documents
            context_docs = await self.gather_context_documents(
                query=query,
                session_id=session_id,
                chat_history=chat_history,
                enable_web_search=should_web_search,
                user_documents=user_documents
            )
            
            # Combine all documents
            all_documents = (
                context_docs["rag_docs"] + 
                context_docs["web_docs"] + 
                context_docs["user_docs"]
            )
            
            # Generate response using RAG chain if documents available
            if all_documents:
                if self.rag_pipeline and context_docs["rag_docs"]:
                    # Use RAG pipeline for response generation
                    response = await self.rag_pipeline.query(
                        question=query,
                        chat_history=chat_history,
                        model=model_override or self.config.default_model,
                        system_prompt_override=system_prompt_override
                    )
                else:
                    # Fallback to direct LLM with context using LCEL
                    rag_chain = self._create_rag_chain(all_documents, query, chat_history, system_prompt_override)
                    response = await rag_chain.ainvoke(query)
            else:
                # No context documents, use direct LLM
                if self.llm_manager:
                    messages = [
                        {"role": "system", "content": system_prompt_override or self.config.default_system_prompt}
                    ]
                    messages.extend(chat_history[-5:])  # Last 5 messages
                    messages.append({"role": "user", "content": query})
                    
                    response = await self.llm_manager.generate_response(
                        messages=messages,
                        model=model_override or self.config.default_model
                    )
                else:
                    response = "I'm sorry, but I'm unable to process your request at the moment."
            
            # Save to memory
            memory.add_user_message(query)
            memory.add_ai_message(response)
            
            # Update session context
            self.session_contexts[session_id]["message_count"] += 2
            
            return {
                "response": response,
                "source": "rag" if all_documents else "llm",
                "session_id": session_id,
                "intent_analysis": intent_analysis,
                "document_counts": {
                    "rag_docs": len(context_docs["rag_docs"]),
                    "web_docs": len(context_docs["web_docs"]),
                    "user_docs": len(context_docs["user_docs"])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "source": "error",
                "session_id": session_id,
                "error": str(e)
            }
    
    async def _generate_greeting_response_stream(self, query: str, chat_history: List[Dict[str, str]], model_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate streaming greeting response using direct LLM calls (reference from rag.py)
        """
        try:
            current_model = model_override or self.config.default_model
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create greeting-specific system prompt
            greeting_system_prompt = """You are a friendly and helpful AI assistant. Respond to greetings and casual conversations naturally and warmly. 
Be conversational, friendly, and concise. Vary your responses to feel natural and human-like. 
Use appropriate emojis sparingly and end with an open question or invitation to chat."""
            
            # Build messages for context
            messages = [{"role": "system", "content": greeting_system_prompt}]
            
            # Include recent chat history for context (last 4 messages max)
            if chat_history:
                recent_history = chat_history[-4:]
                messages.extend(recent_history)
            
            # Add current greeting
            messages.append({"role": "user", "content": f"Current time: {current_time}\n\nUser says: {query}"})
            
            logger.info(f"🤖 Generating greeting response for: '{query}' using model: {current_model}")
            
            # Call LLM based on model type (following rag.py pattern)
            normalized_model = current_model.lower().strip()
            
            # OpenAI models (GPT-4o, GPT-4o-mini, etc.)
            if normalized_model.startswith("gpt-") and self.llm_manager:
                try:
                    # Get OpenAI client from LLM manager if available
                    if hasattr(self.llm_manager, 'openai_client'):
                        openai_client = self.llm_manager.openai_client
                        
                        # Validate the model name
                        valid_openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
                        openai_model_name = normalized_model if normalized_model in valid_openai_models else "gpt-4o-mini"
                        
                        response_stream = await openai_client.chat.completions.create(
                            model=openai_model_name,
                            messages=messages,
                            temperature=0.7,  # Higher temperature for varied responses
                            max_tokens=150,   # Keep greetings concise
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                yield content_piece
                        return
                except Exception as e:
                    logger.error(f"OpenAI greeting streaming error: {e}")
            
            # Claude models - if available through LLM manager
            elif normalized_model.startswith("claude") and self.llm_manager:
                try:
                    if hasattr(self.llm_manager, 'anthropic_client'):
                        anthropic_client = self.llm_manager.anthropic_client
                        
                        # Convert messages to Claude format
                        claude_messages = []
                        system_content = self.config.default_system_prompt
                        
                        for msg in messages:
                            if msg["role"] == "system":
                                system_content = msg["content"]
                            elif msg["role"] != "system":
                                claude_messages.append(msg)

                        # Use the provider's mapping function to get the correct model ID
                        claude_provider = self.llm_manager.providers.get("claude")
                        if not claude_provider:
                             raise ValueError("Claude provider not found in LLM Manager")
                        
                        correct_model_id = claude_provider._map_model_name(current_model)

                        response_stream = await anthropic_client.messages.create(
                            model=correct_model_id, #<-- FIXED: Use the dynamically determined model
                            system=system_content,
                            messages=claude_messages,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            if chunk.type == "content_block_delta" and chunk.delta.text:
                                yield chunk.delta.text
                        return
                except Exception as e:
                    logger.error(f"Claude streaming error: {e}")
            
            # Fallback: Use LLM manager's generate_response method (non-streaming)
            if self.llm_manager:
                try:
                    response = await self.llm_manager.generate_response(
                        messages=messages,
                        model=current_model,
                        temperature=0.7,
                        max_tokens=150
                    )
                    
                    # Simulate streaming by yielding words
                    words = response.split()
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words) - 1 else "")
                        if i % 3 == 0:  # Small delay every few words
                            await asyncio.sleep(0.05)
                    return
                except Exception as e:
                    logger.error(f"LLM manager fallback error: {e}")
            
            # Final fallback responses
            fallback_responses = [
                "Hello! 😊 I'm here and ready to help with whatever you need. What's on your mind today?",
                "Hi there! 👋 Great to see you! What can I assist you with?",
                "Hey! 🤗 I'm doing well and excited to help. What would you like to explore or discuss?",
                "Hello! ✨ Thanks for asking! I'm here and ready to dive into any questions or topics you have. What interests you today?",
                "Hi! 😄 I'm here and eager to help. Feel free to ask me anything or just chat about what's on your mind!"
            ]
            
            # Use query hash for consistent but varied responses
            response_idx = hash(query + str(len(chat_history))) % len(fallback_responses)
            fallback_response = fallback_responses[response_idx]
            
            # Stream the fallback response
            words = fallback_response.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                if i % 3 == 0:  # Small delay every few words
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            logger.error(f"Error in greeting response generation: {e}")
            # Final fallback
            simple_response = "Hello! 😊 I'm here to help. What can I assist you with today?"
            words = simple_response.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                if i % 2 == 0:
                    await asyncio.sleep(0.05)

    async def _generate_llm_response_stream(
        self, 
        messages: List[Dict[str, str]], 
        model_override: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming LLM response using direct LLM calls (reference from rag.py)
        """
        try:
            current_model = model_override or self.config.default_model
            normalized_model = current_model.lower().strip()
            
            logger.info(f"🤖 Generating LLM response using model: {current_model}")
            
            # OpenAI models
            if normalized_model.startswith("gpt-") and self.llm_manager:
                try:
                    if hasattr(self.llm_manager, 'openai_client'):
                        openai_client = self.llm_manager.openai_client
                        
                        valid_openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
                        openai_model_name = normalized_model if normalized_model in valid_openai_models else "gpt-4o-mini"
                        
                        response_stream = await openai_client.chat.completions.create(
                            model=openai_model_name,
                            messages=messages,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                yield content_piece
                        return
                except Exception as e:
                    logger.error(f"OpenAI streaming error: {e}")
            
            # Claude models
            elif normalized_model.startswith("claude") and self.llm_manager:
                try:
                    if hasattr(self.llm_manager, 'anthropic_client'):
                        anthropic_client = self.llm_manager.anthropic_client
                        
                        # Convert messages to Claude format
                        claude_messages = []
                        system_content = self.config.default_system_prompt
                        
                        for msg in messages:
                            if msg["role"] == "system":
                                system_content = msg["content"]
                            elif msg["role"] != "system":
                                claude_messages.append(msg)

                        # Use the provider's mapping function to get the correct model ID
                        claude_provider = self.llm_manager.providers.get("claude")
                        if not claude_provider:
                             raise ValueError("Claude provider not found in LLM Manager")
                        
                        correct_model_id = claude_provider._map_model_name(current_model)

                        response_stream = await anthropic_client.messages.create(
                            model=correct_model_id, #<-- FIXED: Use the dynamically determined model
                            system=system_content,
                            messages=claude_messages,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            if chunk.type == "content_block_delta" and chunk.delta.text:
                                yield chunk.delta.text
                        return
                except Exception as e:
                    logger.error(f"Claude streaming error: {e}")
            
            # Fallback: Use LLM manager's generate_response method (non-streaming)
            if self.llm_manager:
                try:
                    response = await self.llm_manager.generate_response(
                        messages=messages,
                        model=current_model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    # Simulate streaming by yielding words
                    words = response.split()
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words) - 1 else "")
                        if i % 5 == 0:  # Small delay every few words
                            await asyncio.sleep(0.05)
                    return
                except Exception as e:
                    logger.error(f"LLM manager fallback error: {e}")
            
            # Final fallback
            yield "I apologize, but I'm unable to process your request at the moment. Please try again."
                    
        except Exception as e:
            logger.error(f"Error in LLM response generation: {e}")
            yield f"I encountered an error while processing your request: {str(e)}"

    @traceable(name="chatbot_query_stream") if LANGSMITH_AVAILABLE else lambda f: f
    async def query_stream(
        self, 
        session_id: str, 
        query: str,
        enable_web_search: bool = True,
        user_documents: Optional[List[str]] = None,
        model_override: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
        mcp_enabled: bool = False,
        mcp_schema: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query with a corrected, unified streaming logic...
        """
        try:
            self._cleanup_expired_sessions()
            memory = self._get_session_memory(session_id)
            chat_history = self._format_chat_history(memory)
            
            intent_analysis = await self.analyze_query_intent(query, chat_history)
            yield {"type": "analysis", "data": intent_analysis}

            # Add this new block at the beginning to handle the document listing query first.
            if intent_analysis.get("is_document_listing_query"):
                yield {"type": "status", "data": "Checking my knowledge base for available documents..."}
                if self.rag_pipeline:
                    document_sources = await self.rag_pipeline.list_available_documents()
                    if document_sources:
                        response_lines = ["🏷️ **Available Documents**", "\nI have access to the following documents in this GPT's knowledge base:\n"]
                        for i, source in enumerate(document_sources):
                            # Make the source name more readable
                            display_name = os.path.basename(urlparse(source).path)
                            response_lines.append(f"• **Document {i+1}:** {display_name}")
                        final_response = "\n".join(response_lines)
                    else:
                        final_response = "🏷️ **No Documents Found**\n\nMy knowledge base for this GPT is currently empty. You can add documents to build it up!"
                    
                    yield {"type": "content", "data": final_response}
                    memory.add_user_message(query)
                    memory.add_ai_message(final_response)
                    yield {"type": "done", "data": {"source": "introspection"}}
                    return # End the request here.

            
            # PRIORITY 1: FAST GREETING RESPONSE (bypass everything else)
            if intent_analysis.get("is_greeting", False):
                try:
                    # ... (greeting logic remains the same) ...
                    yield {"type": "status", "data": "Generating friendly greeting response..."}
                    response_parts = []
                    async for response_chunk in self._generate_greeting_response_stream(query, chat_history, model_override):
                        response_parts.append(response_chunk)
                        yield {"type": "content", "data": response_chunk}
                    final_response = "".join(response_parts)
                    memory.add_user_message(query)
                    memory.add_ai_message(final_response)
                    yield {"type": "done", "data": {"source": "greeting", "session_id": session_id}}
                    return
                except Exception as e:
                    logger.error(f"❌ Greeting response failed: {e}")
                    # Continue with regular processing
            
            # PRIORITY 2: MCP PROCESSING WITH ENHANCED FALLBACK CASCADE
            if intent_analysis.get("mcp_enabled", False) and mcp_schema:
                yield {"type": "status", "data": f"Processing MCP request for server: {intent_analysis.get('mcp_server', 'auto')}..."}
                
                mcp_success = False
                response_parts = []
                
                # STEP 1: Try MCP first
                try:
                    # Use cleaned query for MCP processing
                    mcp_query = intent_analysis.get("cleaned_query", query)
                    
                    async for response in self.mcp_core.execute_mcp_request(
                        query=mcp_query,
                        mcp_schema=mcp_schema,
                        chat_history=chat_history,
                        api_keys=api_keys,
                        detected_server_name=intent_analysis.get("mcp_server"),
                        model_override=model_override
                    ):
                        if response.get("type") == "content":
                            content = response.get("data", "")
                            response_parts.append(content)
                            yield {"type": "content", "data": content}
                        elif response.get("type") == "done":
                            mcp_success = True
                            break
                        elif response.get("type") == "error":
                            logger.warning(f"MCP returned error: {response.get('data', 'Unknown error')}")
                            break
                    
                    if mcp_success and response_parts:
                        final_response = "".join(response_parts)
                        memory.add_user_message(query)
                        memory.add_ai_message(final_response)
                        yield {"type": "done", "data": {"source": "mcp", "session_id": session_id}}
                        return
                        
                except Exception as e:
                    logger.error(f"❌ MCP request failed: {e}")
                
                # STEP 2: MCP failed, try WebSearch fallback
                if not mcp_success:
                    yield {"type": "status", "data": "🔄 MCP failed, trying web search fallback..."}
                    
                    web_success = False
                    web_docs = []
                    
                    try:
                        if self.web_search and self.web_search.is_available():
                            web_docs = await self.web_search.search(query, self.config.web_search_max_results)
                            
                            if web_docs:
                                logger.info(f"🌐 WebSearch fallback: Retrieved {len(web_docs)} documents")
                                
                                # Generate response using web documents
                                if web_docs:
                                    rag_chain = self._create_rag_chain(web_docs, query, chat_history, system_prompt_override)
                                    response = await rag_chain.ainvoke(query)
                                    
                                    yield {"type": "content", "data": response}
                                    memory.add_user_message(query)
                                    memory.add_ai_message(response)
                                    
                                    yield {"type": "done", "data": {
                                        "source": "websearch_fallback", 
                                        "session_id": session_id,
                                        "fallback_chain": "mcp→websearch"
                                    }}
                                    return
                                    
                    except Exception as e:
                        logger.error(f"❌ WebSearch fallback failed: {e}")
                
                # STEP 3: WebSearch failed, try RAG/Knowledge Base fallback
                if not web_success:
                    yield {"type": "status", "data": "🔄 WebSearch failed, trying knowledge base fallback..."}
                    
                    rag_success = False
                    
                    try:
                        # Gather RAG documents from knowledge base
                        context_docs = await self.gather_context_documents(
                            query=query,
                            session_id=session_id,
                            chat_history=chat_history,
                            enable_web_search=False,  # Don't do web search since it failed
                            user_documents=user_documents
                        )
                        
                        rag_docs = context_docs["rag_docs"]
                        user_docs = context_docs["user_docs"]
                        all_rag_docs = rag_docs + user_docs
                        
                        if all_rag_docs:
                            logger.info(f"📚 RAG fallback: Retrieved {len(all_rag_docs)} documents from knowledge base")
                            
                            # Use RAG pipeline if available, otherwise manual chain
                            if self.rag_pipeline and rag_docs:
                                try:
                                    response = await self.rag_pipeline.query(
                                        question=query,
                                        chat_history=chat_history,
                                        model=model_override or self.config.default_model,
                                        system_prompt_override=system_prompt_override
                                    )
                                    rag_success = True
                                except Exception as e:
                                    logger.error(f"❌ RAG pipeline failed: {e}")
                                    # Try manual chain
                                    rag_chain = self._create_rag_chain(all_rag_docs, query, chat_history, system_prompt_override)
                                    response = await rag_chain.ainvoke(query)
                                    rag_success = True
                            else:
                                # Use manual RAG chain
                                rag_chain = self._create_rag_chain(all_rag_docs, query, chat_history, system_prompt_override)
                                response = await rag_chain.ainvoke(query)
                            
                            if rag_success:
                                yield {"type": "content", "data": response}
                                memory.add_user_message(query)
                                memory.add_ai_message(response)
                                
                                yield {"type": "done", "data": {
                                    "source": "rag_fallback", 
                                    "session_id": session_id,
                                    "fallback_chain": "mcp→websearch→rag",
                                    "document_counts": {
                                        "rag_docs": len(rag_docs),
                                        "user_docs": len(user_docs)
                                    }
                                }}
                                return
                                
                    except Exception as e:
                        logger.error(f"❌ RAG fallback failed: {e}")
                
                # STEP 4: All fallbacks failed, use General LLM response
                yield {"type": "status", "data": "🔄 All fallbacks failed, using general LLM response..."}
                
                try:
                    messages = [
                        {"role": "system", "content": system_prompt_override or self.config.default_system_prompt}
                    ]
                    messages.extend(chat_history[-5:])  # Last 5 messages for context
                    messages.append({"role": "user", "content": query})
                    
                    response_parts = []
                    async for response_chunk in self._generate_llm_response_stream(
                        messages=messages,
                        model_override=model_override
                    ):
                        response_parts.append(response_chunk)
                        yield {"type": "content", "data": response_chunk}
                    
                    final_response = "".join(response_parts)
                    memory.add_user_message(query)
                    memory.add_ai_message(final_response)
                    
                    yield {"type": "done", "data": {
                        "source": "llm_fallback", 
                        "session_id": session_id,
                        "fallback_chain": "mcp→websearch→rag→llm"
                    }}
                    return
                    
                except Exception as e:
                    logger.error(f"❌ Even LLM fallback failed: {e}")
                    yield {"type": "error", "data": f"All processing methods failed: {str(e)}"}
                    pass
            
            # PRIORITY 3: INTELLIGENT RAG + WEB SEARCH DECISION LOGIC
            yield {"type": "status", "data": "Gathering context from all sources..."}

            # Step 1: Gather documents from all sources (user, knowledge base, web)
            should_web_search = enable_web_search and intent_analysis.get("web_search_needed", False)
            context_docs = await self.gather_context_documents(
                query=query,
                session_id=session_id,
                chat_history=chat_history,
                enable_web_search=should_web_search,
                user_documents=user_documents
            )

            # Step 2: Combine all retrieved documents into a single list
            all_documents = (
                context_docs["user_docs"] + 
                context_docs["rag_docs"] + 
                context_docs["web_docs"]
            )
            
            source_type = "llm"
            if context_docs["user_docs"]: source_type = "user_docs"
            elif context_docs["rag_docs"]: source_type = "rag"
            if context_docs["web_docs"]: source_type = "combined" if source_type != "llm" else "web_search"

            # Step 3: Use a single, unified path for generating the response
            response_parts = []
            
            if all_documents:
                # If ANY documents are found, use the streaming RAG pipeline
                yield {"type": "status", "data": f"Generating response using {len(all_documents)} documents from '{source_type}'..."}
                
                if self.rag_pipeline:
                    async for chunk in self.rag_pipeline.query_stream(
                        question=query,
                        chat_history=chat_history,
                        model=model_override or self.config.default_model,
                        pre_retrieved_docs=all_documents, # Pass all found docs here
                        system_prompt_override=system_prompt_override
                    ):
                        response_parts.append(chunk)
                        yield {"type": "content", "data": chunk}
                else:
                    # Fallback if RAG pipeline isn't available for some reason
                    yield {"type": "error", "data": "RAG Pipeline is not available."}

            else:
                # If NO documents are found, use the direct LLM stream
                yield {"type": "status", "data": "No relevant documents found. Using general knowledge."}
                messages = [{"role": "system", "content": system_prompt_override or self.config.default_system_prompt}]
                messages.extend(chat_history[-5:])
                messages.append({"role": "user", "content": query})
                
                async for response_chunk in self._generate_llm_response_stream(
                    messages=messages, model_override=model_override
                ):
                    response_parts.append(response_chunk)
                    yield {"type": "content", "data": response_chunk}

            # Step 4: Finalize and save to memory
            final_response = "".join(response_parts)
            memory.add_user_message(query)
            memory.add_ai_message(final_response)
            self.session_contexts[session_id]["message_count"] += 2

            yield {"type": "done", "data": {
                "source": source_type,
                "session_id": session_id,
                "document_counts": {
                    "rag_docs": len(context_docs["rag_docs"]),
                    "web_docs": len(context_docs["web_docs"]),
                    "user_docs": len(context_docs["user_docs"]),
                    "total_chosen": len(all_documents)
                }
            }}
            
        except Exception as e:
            logger.error(f"❌ Error in query stream: {e}")
            yield {"type": "error", "data": f"Error processing query: {str(e)}"}
    
    async def add_documents_to_knowledge_base(
        self,
        source_urls: List[str]
    ) -> Dict[str, Any]:
        """
        Adds documents from a list of URLs to the knowledge base using the
        new "Fetch First" production-grade workflow.
        """
        if not self.rag_pipeline:
            return {"success": False, "error": "RAG pipeline not available"}
        if not self.storage_client:
            return {"success": False, "error": "Storage client not configured"}

        logger.info(f"Starting ingestion for {len(source_urls)} URLs.")
        
        successfully_stored_urls = []
        failed_urls = []

        # Step 1 & 2: Iterate and process each URL using the robust storage client
        for url in source_urls:
            logger.info(f"Processing URL: {url}")
            # Use the new robust method from the storage client
            success, stored_url_or_error = self.storage_client.download_file_from_url(
                url=url, 
                is_user_doc=False # These are for the knowledge base
            )
            
            if success:
                logger.info(f"✅ Successfully stored content from {url} at {stored_url_or_error}")
                successfully_stored_urls.append(stored_url_or_error)
            else:
                logger.error(f"❌ Failed to process URL {url}: {stored_url_or_error}")
                failed_urls.append({"url": url, "error": stored_url_or_error})

        # Step 3: Index the successfully stored documents into the RAG pipeline
        if successfully_stored_urls:
            logger.info(f"Indexing {len(successfully_stored_urls)} successfully stored documents...")
            try:
                # The RAG pipeline can now confidently process these URLs
                doc_ids = await self.rag_pipeline.add_documents_from_urls(successfully_stored_urls)
                logger.info(f"✅ Indexing complete for {len(doc_ids)} document chunks.")
                return {
                    "success": True,
                    "processed_count": len(successfully_stored_urls),
                    "indexed_chunk_ids": doc_ids,
                    "failures": failed_urls
                }
            except Exception as e:
                logger.critical(f"CRITICAL: Indexing failed even after successful storage: {e}")
                return {
                    "success": False,
                    "error": f"Indexing failed: {str(e)}",
                    "processed_count": len(successfully_stored_urls),
                    "failures": failed_urls
                }
        else:
            logger.warning("No URLs were successfully processed.")
            return {
                "success": False,
                "error": "No documents could be processed from the provided URLs.",
                "processed_count": 0,
                "failures": failed_urls
            }
            
    # This method can now be a simple alias, as user docs and KB docs share the same robust pipeline
    async def add_user_documents_for_session(self, session_id: str, urls: List[str]) -> Dict[str, Any]:
        """
        Adds user-provided documents for a specific session by indexing them.
        This now uses the same robust pipeline as the main knowledge base.
        """
        logger.info(f"Adding {len(urls)} user documents for session {session_id}.")
        # The underlying workflow is now identical, we just call the main method.
        # The storage layer will correctly label them as `is_user_doc=True` if needed,
        # but for now, they share the same ingestion logic.
        return await self.add_documents_to_knowledge_base(urls)


    
    async def add_documents_from_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Add documents from URLs to the knowledge base with R2 storage integration - FIXED"""
        try:
            if not self.rag_pipeline:
                return {"success": False, "error": "RAG pipeline not available"}
            
            uploaded_files = []
            
            # FIXED: Don't pre-process R2 URLs - let RAG pipeline handle them directly
            if self.storage_client:
                for url in urls:
                    try:
                        # Only pre-process external URLs, not our own R2 URLs
                        parsed_url = urlparse(url) 
                        is_our_r2_url = (
                            self.storage_client.account_id and 
                            self.storage_client.bucket_name and 
                            f"{self.storage_client.bucket_name}.{self.storage_client.account_id}.r2.cloudflarestorage.com" in parsed_url.netloc
                        )
                        
                        if not is_our_r2_url:
                            # Only download and store external URLs
                            success, file_url_or_error = self.storage_client.download_file_from_url(url)
                            if success:
                                uploaded_files.append(file_url_or_error)
                                logger.info(f"✅ Downloaded and stored external URL: {url}")
                            else:
                                logger.error(f"❌ Failed to download external URL {url}: {file_url_or_error}")
                        else:
                            logger.info(f"🔍 Detected R2 URL, will be handled directly by RAG pipeline: {url}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing URL {url}: {e}")
            
            # Add to RAG pipeline (this will now handle R2 URLs correctly)
            doc_ids = await self.rag_pipeline.add_documents_from_urls(urls)
            
            logger.info(f"✅ Added documents from {len(urls)} URLs")
            
            return {
                "success": True,
                "document_ids": doc_ids,
                "uploaded_files": uploaded_files,
                "url_count": len(urls),
                "r2_storage_used": self.storage_client is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Error adding documents from URLs: {e}")
            return {"success": False, "error": str(e)}

    async def add_user_documents_for_session(self, session_id: str, urls: List[str]) -> Dict[str, Any]:
        """
        FIXED: Adds user-provided documents for a specific session by indexing them into the knowledge base.
        This is an alias for add_documents_from_urls, as the current architecture uses a shared KB
        for all documents associated with a given agent instance.
        """
        logger.info(f"Adding {len(urls)} user documents for session {session_id} to the knowledge base.")
        # The current architecture adds all documents to the same knowledge base.
        # This method is implemented to resolve the AttributeError from main_app.py
        # and uses the existing functionality for URL-based document indexing.
        return await self.add_documents_from_urls(urls)
    
    async def add_documents_from_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents from files to the knowledge base with R2 storage integration"""
        try:
            if not self.rag_pipeline:
                return {"success": False, "error": "RAG pipeline not available"}
            
            uploaded_files = []
            
            # Upload files to R2 storage if available
            if self.storage_client:
                for file_path in file_paths:
                    try:
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        filename = os.path.basename(file_path)
                        success, file_url = self.storage_client.upload_file(
                            file_data=file_data,
                            filename=filename,
                            is_user_doc=False,  # Knowledge base document
                            schedule_deletion_hours=self.config.file_deletion_hours
                        )
                        
                        if success:
                            uploaded_files.append(file_url)
                            logger.info(f"✅ Uploaded file to R2: {file_path}")
                        else:
                            logger.error(f"❌ Failed to upload file: {file_path}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing file {file_path}: {e}")
            
            # Add to RAG pipeline
            doc_ids = await self.rag_pipeline.add_documents_from_files(file_paths)
            
            logger.info(f"✅ Added documents from {len(file_paths)} files")
            
            return {
                "success": True,
                "document_ids": doc_ids,
                "uploaded_files": uploaded_files,
                "file_count": len(file_paths),
                "r2_storage_used": self.storage_client is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Error adding documents from files: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup_expired_storage_files(self) -> Dict[str, Any]:
        """Clean up expired files from R2 storage"""
        try:
            if not self.storage_client:
                return {"success": False, "error": "Storage client not available"}
            
            deleted_count = self.storage_client.check_and_delete_expired_files()
            
            logger.info(f"🧹 Cleaned up {deleted_count} expired files from storage")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Cleaned up {deleted_count} expired files"
            }
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up storage files: {e}")
            return {"success": False, "error": str(e)}
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get storage system status"""
        if not self.storage_client:
            return {
                "available": False,
                "reason": "Storage client not initialized"
            }
        
        return {
            "available": True,
            "r2_enabled": not self.storage_client.use_local_fallback,
            "local_fallback": self.storage_client.use_local_fallback,
            "bucket_name": getattr(self.storage_client, 'bucket_name', 'unknown'),
            "account_id": getattr(self.storage_client, 'account_id', 'unknown')
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and availability including storage"""
        return {
            "llm_manager": self.llm_manager is not None,
            "rag_pipeline": self.rag_pipeline is not None,
            "web_search": self.web_search is not None and self.web_search.is_available(),
            "mcp_core": self.mcp_core is not None,
            "storage_client": self.storage_client is not None,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "storage_available": STORAGE_AVAILABLE,
            "active_sessions": len(self.sessions),
            "available_models": self.llm_manager.get_available_providers() if self.llm_manager else [],
            "storage_status": self.get_storage_status()
        }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session"""
        if session_id not in self.session_contexts:
            return {"exists": False}
        
        context = self.session_contexts[session_id]
        memory = self.sessions.get(session_id)
        
        return {
            "exists": True,
            "created_at": context.get("created_at"),
            "last_activity": context.get("last_activity"),
            "message_count": context.get("message_count", 0),
            "memory_length": len(memory.messages) if memory else 0
        }

    async def cleanup(self):
        """Cleanup all connections and resources"""
        logger.info("🧹 Starting ChatbotAgent cleanup...")
        
        try:
            # Cleanup HTTP clients
            if hasattr(self, 'llm_manager') and self.llm_manager:
                await self.llm_manager.cleanup()
            
            # Cleanup RAG pipeline
            if hasattr(self, 'rag_pipeline') and self.rag_pipeline:
                await self.rag_pipeline.cleanup()
            
            # Cleanup web search
            if hasattr(self, 'web_search') and self.web_search:
                await self.web_search.cleanup()
            
            # Cleanup MCP core
            if hasattr(self, 'mcp_core') and self.mcp_core:
                await self.mcp_core.cleanup_processes()
            
            # Cleanup storage client
            if hasattr(self, 'storage_client') and self.storage_client:
                if hasattr(self.storage_client, 'cleanup'):
                    await self.storage_client.cleanup()
            
            # Clear sessions
            self.sessions.clear()
            self.session_contexts.clear()
            
            logger.info("✅ ChatbotAgent cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during ChatbotAgent cleanup: {e}")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

# Convenience functions for easy usage
async def create_chatbot_agent(config: Optional[ChatbotConfig] = None, storage_client: Optional['CloudflareR2Storage'] = None) -> ChatbotAgent:
    """Create and initialize a chatbot agent with storage integration"""
    agent = ChatbotAgent(config, storage_client=storage_client)
    await agent.initialize()
    return agent

if __name__ == "__main__":
    asyncio.run()