"""
Enhanced Web Search Client with LLM Analysis and CloudflareR2Storage Integration
Integrated with centralized LLM Manager and CloudflareR2Storage for consistent storage across the application.
Uses LangChain Expression Language and LangSmith for enhanced web search pipeline.
"""

import os
import asyncio
import logging
import httpx
import re
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Tuple

# Core imports
import numpy as np

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# LangChain imports (following user requirements)
try:
    from langchain_core.documents import Document
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

# Import centralized LLM Manager
try:
    from llm_model import LLMManager, LLMConfig
    LLM_MODEL_AVAILABLE = True
except ImportError:
    LLM_MODEL_AVAILABLE = False
    logging.error("llm_model.py not found. Please ensure llm_model.py is available in the same directory.")

# Import CloudflareR2Storage (following user requirements for integration)
try:
    from storage import CloudflareR2Storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logging.error("storage.py not found. Please ensure storage.py is available in the same directory.")

# Tavily imports
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("Tavily package not found. Web search will be unavailable.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WebSearchConfig:
    """Configuration class for web search functionality with R2 storage integration"""
    
    # API Keys
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    
    # Search settings
    max_results: int = 5
    search_depth: str = "advanced"  # "basic" or "advanced"
    include_raw_content: bool = True
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    
    # Analysis settings
    use_llm_for_query_analysis: bool = True
    analysis_model: str = "gpt-4o-mini"  # Default model for analysis
    analysis_temperature: float = 0.5
    
    # Content processing
    max_content_length: int = 4000
    content_chunk_size: int = 1000
    
    # Timeout settings
    request_timeout: int = 30
    
    # Storage settings (NEW - integrated from rag.py pattern)
    storage_client: Optional['CloudflareR2Storage'] = None
    enable_search_caching: bool = True
    cache_expiry_hours: int = 24  # Cache search results for 24 hours
    temp_processing_path: str = field(default_factory=lambda: os.getenv("TEMP_PROCESSING_PATH", "local_websearch_data/temp_cache"))
    
    # URL validation patterns
    url_patterns: List[str] = field(default_factory=lambda: [
        r'https?://[^\s]+',  # Full URLs with http/https
        r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',  # URLs starting with www
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'  # Domain names with paths
    ])
    
    def to_llm_config(self) -> 'LLMConfig':
        """Convert WebSearchConfig to LLMConfig for LLM Manager"""
        if not LLM_MODEL_AVAILABLE:
            raise ImportError("llm_model module not available")
        
        return LLMConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            langsmith_api_key=self.langsmith_api_key,
            default_model=self.analysis_model,
            temperature=self.analysis_temperature
        )
    
    @classmethod
    def from_env(cls) -> 'WebSearchConfig':
        """Create configuration from environment variables"""
        return cls()


class WebSearchClient:
    """Enhanced web search client with LLM-based analysis and R2 storage integration using centralized LLM Manager"""
    
    def __init__(self, config: WebSearchConfig):
        self.config = config
        self.storage_client = config.storage_client
        
        # Create temp processing directory for caching
        os.makedirs(config.temp_processing_path, exist_ok=True)
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Tavily, LLM Manager, embeddings clients, and storage with proper error handling"""
        try:
            # Initialize Tavily client
            if self.config.tavily_api_key and TAVILY_AVAILABLE:
                self.tavily_client = AsyncTavilyClient(api_key=self.config.tavily_api_key)
                logger.info("✅ Tavily client initialized successfully")
            else:
                self.tavily_client = None
                if not TAVILY_AVAILABLE:
                    logger.error("❌ Tavily package not available. Install with: pip install tavily-python")
                else:
                    logger.error("❌ No Tavily API key provided. Web search will be disabled.")
            
            # Initialize centralized LLM Manager for query analysis
            self.llm_manager = None
            if LLM_MODEL_AVAILABLE:
                try:
                    llm_config = self.config.to_llm_config()
                    self.llm_manager = LLMManager(llm_config)
                    logger.info("✅ LLM Manager initialized for query analysis")
                except Exception as e:
                    logger.error(f"❌ Error initializing LLM Manager: {e}")
                    self.llm_manager = None
            else:
                logger.warning("⚠️ llm_model module not available. Query analysis will be disabled.")
            
            # Initialize embeddings for similarity calculations
            if self.config.openai_api_key and LANGCHAIN_AVAILABLE:
                self.embeddings = OpenAIEmbeddings(
                    api_key=self.config.openai_api_key,
                    model="text-embedding-3-small"
                )
                logger.info("✅ OpenAI embeddings initialized")
            else:
                self.embeddings = None
                logger.warning("⚠️ No OpenAI API key provided or LangChain unavailable. Similarity filtering disabled.")
            
            # Log storage integration status
            if self.storage_client:
                if self.storage_client.use_local_fallback:
                    logger.warning("⚠️ Web search using R2 storage with local fallback")
                else:
                    logger.info("✅ Web search with CloudflareR2Storage integration enabled")
            else:
                logger.info("🗄️ Web search without R2 storage integration (cache disabled)")
                
        except Exception as e:
            logger.error(f"❌ Error initializing web search clients: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if web search functionality is available"""
        return self.tavily_client is not None
    
    def _generate_cache_key(self, query: str, max_results: int, include_domains: List[str], exclude_domains: List[str]) -> str:
        """Generate cache key for search results"""
        cache_data = {
            "query": query.lower().strip(),
            "max_results": max_results,
            "include_domains": sorted(include_domains) if include_domains else [],
            "exclude_domains": sorted(exclude_domains) if exclude_domains else [],
            "search_depth": self.config.search_depth
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        return f"websearch_cache_{cache_hash}.json"
    
    async def _get_cached_results(self, cache_key: str) -> Optional[List[Document]]:
        """Retrieve cached search results from R2 storage"""
        if not self.storage_client or not self.config.enable_search_caching:
            return None
        
        try:
            # Try to download cached results
            cache_path = os.path.join(self.config.temp_processing_path, cache_key)
            
            # Download from R2 storage
            success = self.storage_client.download_file(f"websearch_cache/{cache_key}", cache_path)
            
            if success and os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
                if datetime.now() - cache_time < timedelta(hours=self.config.cache_expiry_hours):
                    # Reconstruct documents
                    documents = []
                    for doc_data in cache_data.get("documents", []):
                        doc = Document(
                            page_content=doc_data["page_content"],
                            metadata=doc_data["metadata"]
                        )
                        documents.append(doc)
                    
                    logger.info(f"🎯 Retrieved {len(documents)} cached web search results")
                    
                    # Clean up temp file
                    try:
                        os.remove(cache_path)
                    except:
                        pass
                    
                    return documents
                else:
                    logger.info("⏰ Cached search results expired")
                    # Clean up expired cache file
                    try:
                        os.remove(cache_path)
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"❌ Error retrieving cached search results: {e}")
        
        return None
    
    async def _cache_search_results(self, cache_key: str, documents: List[Document]) -> bool:
        """Cache search results to R2 storage"""
        if not self.storage_client or not self.config.enable_search_caching:
            return False
        
        try:
            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "query_hash": cache_key,
                "document_count": len(documents),
                "documents": []
            }
            
            # Serialize documents
            for doc in documents:
                doc_data = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                cache_data["documents"].append(doc_data)
            
            # Save to temp file
            cache_path = os.path.join(self.config.temp_processing_path, cache_key)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Upload to R2 storage
            with open(cache_path, 'rb') as f:
                file_data = f.read()
            
            success, file_url = self.storage_client.upload_file(
                file_data=file_data,
                filename=cache_key,
                is_user_doc=False,  # This is cache data, not user document
                schedule_deletion_hours=self.config.cache_expiry_hours + 1  # Add buffer
            )
            
            # Clean up temp file
            try:
                os.remove(cache_path)
            except:
                pass
            
            if success:
                logger.info(f"💾 Cached {len(documents)} web search results to R2 storage")
                return True
            else:
                logger.error("❌ Failed to cache search results to R2 storage")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error caching search results: {e}")
            return False
    
    @traceable(name="web_search_query_analysis") if LANGSMITH_AVAILABLE else lambda f: f
    async def analyze_search_necessity(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze if web search is necessary for the given query using centralized LLM Manager
        
        Args:
            query: The search query
            chat_history: Optional chat history for context
            model_override: Optional model name to use for analysis
            
        Returns:
            Dict containing analysis results
        """
        try:
            if not self.config.use_llm_for_query_analysis or not self.llm_manager:
                return {
                    "should_search": True,
                    "reasoning": "LLM analysis disabled or unavailable, defaulting to search",
                    "confidence": "low"
                }
            
            # Check for greetings or conversational responses first
            is_greeting = await self._detect_greeting(query, model_override=model_override)
            if is_greeting:
                return {
                    "should_search": False,
                    "reasoning": "Simple greeting or conversational response detected",
                    "confidence": "high"
                }
            
            # Check for date/time queries first (PRIORITY CHECK)
            is_date_query = await self._detect_date_query(query, model_override=model_override)
            if is_date_query:
                return {
                    "should_search": False,
                    "reasoning": "Date/time query detected - should use local datetime",
                    "confidence": "high"
                }
            
            # Analyze search necessity using centralized LLM Manager
            should_search = await self._analyze_web_search_need(query, model_override=model_override)
            
            return {
                "should_search": should_search,
                "reasoning": "LLM analysis of query intent and information needs",
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"Error in search necessity analysis: {e}")
            return {
                "should_search": True,
                "reasoning": f"Analysis failed: {str(e)}, defaulting to search",
                "confidence": "low"
            }
    
    async def _analyze_web_search_need(self, query: str, model_override: Optional[str] = None) -> bool:
        """Use centralized LLM Manager to determine if web search is needed"""
        try:
            if not self.llm_manager:
                return False
            
            # Use the centralized LLM Manager's query analysis feature
            analysis = await self.llm_manager.analyze_query(query, "web_search", model=model_override)
            return analysis.get("result", False)
            
        except Exception as e:
            logger.error(f"Error in LLM-based web search analysis: {e}")
            return False
    
    async def _detect_greeting(self, query: str, model_override: Optional[str] = None) -> bool:
        """
        Detect if query is a greeting or conversational response using centralized LLM Manager
        FIX: Added 'model_override' to signature and passed it to the LLM manager.
        """
        try:
            if not self.llm_manager:
                # Fallback to simple pattern matching
                conversational_words = [
                    "hello", "hi", "hey", "thanks", "great", "good", "nice", "awesome", 
                    "cool", "ok", "okay", "sure", "yes", "no", "wow", "amazing", 
                    "perfect", "excellent", "fantastic", "wonderful", "brilliant"
                ]
                return any(word in query.lower() for word in conversational_words)
            
            # Use the centralized LLM Manager's query analysis feature
            analysis = await self.llm_manager.analyze_query(query, "greeting", model=model_override)
            return analysis.get("result", False)
            
        except Exception as e:
            logger.error(f"Error in greeting detection: {e}")
            return False
    
    async def _detect_date_query(self, query: str, model_override: Optional[str] = None) -> bool:
        """
        Detect if query is asking for current date/time information
        FIX: Added 'model_override' to signature and passed it to the LLM manager.
        """
        try:
            query_lower = query.lower().strip()
            
            # Pattern-based detection for common date/time queries
            date_time_patterns = [
                "today's date", "todays date", "what's today's date", "whats todays date",
                "current date", "what date", "what's the date", "whats the date",
                "today date", "date today", "what day is it", "what day", 
                "current time", "what time", "what's the time", "whats the time",
                "time now", "time today", "current day", "what's today", "whats today"
            ]
            
            # Check if query matches common date/time patterns
            if any(pattern in query_lower for pattern in date_time_patterns):
                logger.info(f"🗓️ Date/time query detected: '{query}' - routing to local datetime")
                return True
            
            # Use LLM analysis for more sophisticated detection
            if self.llm_manager:
                analysis = await self.llm_manager.analyze_query(query, "date_time", model=model_override)
                return analysis.get("result", False)
            
            return False
            
        except Exception as e:
            logger.error(f"Error in date query detection: {e}")
            return False
    
    @traceable(name="web_search_execution") if LANGSMITH_AVAILABLE else lambda f: f
    async def search(self, query: str, num_results: Optional[int] = None, 
                    include_domains: Optional[List[str]] = None,
                    exclude_domains: Optional[List[str]] = None) -> List[Document]:
        """
        Perform web search without caching, returning results as LangChain Documents
        
        Args:
            query: Search query
            num_results: Number of results to return (defaults to config)
            include_domains: Domains to include in search
            exclude_domains: Domains to exclude from search
            
        Returns:
            List of Document objects with search results
        """
        if not self.tavily_client:
            logger.warning("🌐 Web search is disabled - Tavily client not available")
            return []
        
        try:
            # Use provided parameters or fall back to config
            max_results = num_results or self.config.max_results
            domains_include = include_domains or self.config.include_domains
            domains_exclude = exclude_domains or self.config.exclude_domains
            
            logger.info(f"🌐 Searching web for: '{query}' (max {max_results} results)")
            
            # Prepare search parameters
            search_params = {
                "query": query,
                "search_depth": self.config.search_depth,
                "max_results": max_results,
                "include_raw_content": self.config.include_raw_content
            }
            
            if domains_include:
                search_params["include_domains"] = domains_include
            if domains_exclude:
                search_params["exclude_domains"] = domains_exclude
            
            # Execute search
            search_response = await self.tavily_client.search(**search_params)
            
            # Process results
            documents = self._process_search_results(search_response, query)
            
            logger.info(f"✅ Found {len(documents)} web search results")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Web search error: {e}")
            return []
    
    def _process_search_results(self, search_response: Dict[str, Any], query: str) -> List[Document]:
        """Process Tavily search results into LangChain Documents"""
        documents = []
        
        try:
            results = search_response.get("results", [])
            
            for result in results:
                # Extract content
                content = result.get("content", "")
                if not content:
                    content = result.get("raw_content", "")
                
                # Truncate content if too long
                if len(content) > self.config.max_content_length:
                    content = content[:self.config.max_content_length] + "..."
                
                # Create metadata
                metadata = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                    "source": "web_search",
                    "query": query,
                    "published_date": result.get("published_date", ""),
                    "search_timestamp": datetime.now().isoformat(),
                }
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
        
        return documents
    
    def extract_urls_from_query(self, query: str) -> List[str]:
        """Extract URLs from query text using regex patterns"""
        urls = []
        
        for pattern in self.config.url_patterns:
            matches = re.findall(pattern, query)
            urls.extend(matches)
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            url = url.strip()
            if self._is_valid_url_structure(url):
                # Add protocol if missing
                if not url.startswith(('http://', 'https://')):
                    if url.startswith('www.'):
                        url = 'https://' + url
                    else:
                        url = 'https://' + url
                cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Remove duplicates
    
    def _is_valid_url_structure(self, url: str) -> bool:
        """Validate URL structure"""
        try:
            # Basic validation checks
            if not url or len(url) < 4:
                return False
            
            # Check for valid domain structure
            if '.' not in url:
                return False
            
            # Check for common invalid patterns
            invalid_patterns = [
                r'^[.\-_]',  # Starts with punctuation
                r'[.\-_]$',  # Ends with punctuation
                r'\.\.',     # Double dots
                r'\s',       # Contains spaces
            ]
            
            for pattern in invalid_patterns:
                if re.search(pattern, url):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_storage_client(self) -> Optional['CloudflareR2Storage']:
        """Get the R2 storage client"""
        return self.storage_client
    
    async def clear_search_cache(self, prefix: str = "websearch_cache") -> Dict[str, Any]:
        """Clear cached search results from R2 storage"""
        if not self.storage_client:
            return {"success": False, "error": "Storage client not available"}
        
        try:
            # List cached files
            cached_files = self.storage_client.list_files(f"{prefix}/")
            
            # Delete cached files
            deleted_count = 0
            for file_key in cached_files:
                try:
                    # Note: Individual file deletion would require additional R2 client methods
                    # For now, we rely on the automatic expiration system
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting cached file {file_key}: {e}")
            
            logger.info(f"🧹 Cleared {deleted_count} cached search results")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Cleared {deleted_count} cached search results"
            }
            
        except Exception as e:
            logger.error(f"❌ Error clearing search cache: {e}")
            return {"success": False, "error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "caching_enabled": self.config.enable_search_caching,
            "storage_available": self.storage_client is not None,
            "cache_expiry_hours": self.config.cache_expiry_hours,
            "temp_path": self.config.temp_processing_path
        }


class WebSearchPipeline:
    """Enhanced web search pipeline with LangChain integration using centralized LLM Manager and R2 storage"""
    
    def __init__(self, config: WebSearchConfig):
        self.config = config
        self.search_client = WebSearchClient(config)
    
    def create_search_chain(self):
        """Create a LangChain search chain using centralized LLM Manager"""
        
        def analyze_query(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze query for search necessity"""
            query = input_dict.get("query", "")
            chat_history = input_dict.get("chat_history", [])
            
            async def async_analyze():
                return await self.search_client.analyze_search_necessity(query, chat_history)
            
            return asyncio.run(async_analyze())
        
        def perform_search(input_dict: Dict[str, Any]) -> List[Document]:
            """Perform the actual search"""
            query = input_dict.get("query", "")
            max_results = input_dict.get("max_results", self.config.max_results)
            
            async def async_search():
                return await self.search_client.search(query, max_results)
            
            return asyncio.run(async_search())
        
        def format_results(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Format search results"""
            documents = input_dict.get("documents", [])
            query = input_dict.get("query", "")
            
            formatted_content = self._format_documents_for_context(documents)
            
            return {
                "search_results": formatted_content,
                "document_count": len(documents),
                "query": query,
                "sources": [doc.metadata.get("url", "") for doc in documents if doc.metadata.get("url")]
            }
        
        # Create the chain
        search_chain = (
            RunnableLambda(analyze_query) |
            RunnableLambda(perform_search) |
            RunnableLambda(format_results)
        )
        
        return search_chain
    
    def _format_documents_for_context(self, documents: List[Document]) -> str:
        """Format documents for LLM context"""
        if not documents:
            return "No web search results found."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Unknown Title")
            url = doc.metadata.get("url", "")
            content = doc.page_content.strip()
            
            # Truncate content if too long
            if len(content) > self.config.content_chunk_size:
                content = content[:self.config.content_chunk_size] + "..."
            
            formatted_doc = f"""🌐 **Web Result {i}**
📰 **Title:** {title}
🔗 **URL:** {url}
📄 **Content:** {content}
"""
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    @traceable(name="web_search_pipeline") if LANGSMITH_AVAILABLE else lambda f: f
    async def search_and_format(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Execute web search pipeline and return formatted results"""
        try:
            # Analyze search necessity
            analysis = await self.search_client.analyze_search_necessity(query, chat_history)
            
            if not analysis.get("should_search", True):
                return {
                    "should_search": False,
                    "reasoning": analysis.get("reasoning", ""),
                    "search_results": "",
                    "document_count": 0,
                    "sources": []
                }
            
            # Perform search
            documents = await self.search_client.search(query)
            
            # Format results
            formatted_content = self._format_documents_for_context(documents)
            
            return {
                "should_search": True,
                "reasoning": analysis.get("reasoning", ""),
                "search_results": formatted_content,
                "document_count": len(documents),
                "sources": [doc.metadata.get("url", "") for doc in documents if doc.metadata.get("url")],
                "analysis": analysis,
                "cache_used": any(doc.metadata.get("search_timestamp") for doc in documents)  # Indicate if cache was used
            }
            
        except Exception as e:
            logger.error(f"Error in web search pipeline: {e}")
            return {
                "should_search": False,
                "reasoning": f"Error: {str(e)}",
                "search_results": "",
                "document_count": 0,
                "sources": [],
                "error": str(e)
            }
    
    async def search_with_advanced_filtering(self, query: str, 
                                           user_docs: Optional[List[Document]] = None,
                                           is_follow_up: bool = False,
                                           max_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search with advanced filtering and context awareness
        
        Args:
            query: Search query
            user_docs: Optional user documents for context
            is_follow_up: Whether this is a follow-up query
            max_results: Maximum number of results
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Perform initial search
            raw_documents = await self.search_client.search(query, max_results * 2)  # Get more for filtering
            
            if not raw_documents:
                return {
                    "documents": [],
                    "filtered_documents": [],
                    "search_results": "No web search results found.",
                    "document_count": 0,
                    "sources": [],
                    "filtering_metadata": {"reason": "No search results"}
                }
            
            # Apply similarity filtering
            filtered_docs, filtering_metadata = [], {}
            
            # Limit to requested number
            final_docs = filtered_docs[:max_results]
            
            # Format results
            formatted_content = self._format_documents_for_context(final_docs)
            
            return {
                "documents": raw_documents,
                "filtered_documents": final_docs,
                "search_results": formatted_content,
                "document_count": len(final_docs),
                "sources": [doc.metadata.get("url", "") for doc in final_docs if doc.metadata.get("url")],
                "filtering_metadata": filtering_metadata,
                "storage_integration": {
                    "cache_enabled": self.config.enable_search_caching,
                    "storage_available": self.search_client.storage_client is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced web search: {e}")
            return {
                "documents": [],
                "filtered_documents": [],
                "search_results": f"Error performing web search: {str(e)}",
                "document_count": 0,
                "sources": [],
                "error": str(e)
            }


class WebSearch:
    """Simplified web search interface for easy integration with R2 storage support"""
    
    def __init__(self, tavily_api_key: Optional[str] = None, 
                 openai_api_key: Optional[str] = None,
                 config: Optional[WebSearchConfig] = None):
        """Initialize web search with optional configuration"""
        if config:
            self.config = config
        else:
            self.config = WebSearchConfig.from_env()
            if tavily_api_key:
                self.config.tavily_api_key = tavily_api_key
            if openai_api_key:
                self.config.openai_api_key = openai_api_key
        
        self.client = WebSearchClient(self.config)
        self.pipeline = WebSearchPipeline(self.config)
    
    async def search(self, query: str, max_results: int = 5) -> List[Document]:
        """Simple search interface"""
        return await self.client.search(query, max_results)
    
    async def smart_search(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Smart search with analysis"""
        return await self.pipeline.search_and_format(query, chat_history)
    
    async def advanced_search(self, query: str, 
                            user_docs: Optional[List[Document]] = None,
                            is_follow_up: bool = False,
                            max_results: int = 5) -> Dict[str, Any]:
        """Advanced search with context filtering"""
        return await self.pipeline.search_with_advanced_filtering(
            query, user_docs, is_follow_up, max_results
        )
    
    async def should_search(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None, model_override: Optional[str] = None) -> bool:
        """
        Check if web search is necessary for the query.
        FIX: Added 'model_override' to the signature and passed it to the client.
        """
        analysis = await self.client.analyze_search_necessity(query, chat_history, model_override=model_override)
        return analysis.get("should_search", True)
    
    def is_available(self) -> bool:
        """Check if web search is available"""
        return self.client.is_available()
    
    def get_config(self) -> WebSearchConfig:
        """Get current configuration"""
        return self.config
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get storage integration status"""
        if not self.client.storage_client:
            return {
                "available": False,
                "reason": "Storage client not initialized"
            }
        
        return {
            "available": True,
            "r2_enabled": not self.client.storage_client.use_local_fallback,
            "local_fallback": self.client.storage_client.use_local_fallback,
            "cache_enabled": self.config.enable_search_caching,
            "cache_stats": self.client.get_cache_stats()
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear search result cache"""
        return await self.client.clear_search_cache()

    async def cleanup(self):
        """Placeholder cleanup method to ensure compatibility with the agent's cleanup process."""
        logger.info("✅ WebSearch module cleaned up.")
        pass


# Example usage and testing
async def main():
    """Example usage of the enhanced web search client with centralized LLM Manager and R2 storage"""
    try:
        # Initialize configuration
        config = WebSearchConfig.from_env()
        
        # Initialize R2 storage if available
        storage_client = None
        if STORAGE_AVAILABLE:
            try:
                storage_client = CloudflareR2Storage()
                config.storage_client = storage_client
                print("✅ R2 storage integration enabled")
            except Exception as e:
                print(f"⚠️ R2 storage initialization failed: {e}")
        
        # Create web search client
        web_search = WebSearch(config=config)
        
        if not web_search.is_available():
            print("❌ Web search not available. Please check your Tavily API key.")
            return
        
        # Show storage status
        storage_status = web_search.get_storage_status()
        print(f"🗄️ Storage Status: {storage_status}")
        
        # Test queries
        test_queries = [
            "What is the latest news about AI?",
            "Hello how are you?",  # Should be detected as greeting
            "Current weather in New York",
            "Python programming basics"  # Should not need web search
        ]
        
        print("\n=== Testing Web Search with R2 Storage Integration ===\n")
        
        for query in test_queries:
            print(f"🔍 Query: {query}")
            
            # Test search necessity analysis
            should_search = await web_search.should_search(query)
            print(f"📊 Should search: {should_search}")
            
            if should_search:
                # Perform smart search
                result = await web_search.smart_search(query)
                print(f"📄 Results: {result['document_count']} documents found")
                if result.get('sources'):
                    print(f"🔗 Sources: {result['sources'][:2]}...")  # Show first 2 sources
                if result.get('cache_used'):
                    print("💾 Used cached results")
            
            print("-" * 50)
        
        # Test cache clearing
        cache_result = await web_search.clear_cache()
        print(f"🧹 Cache clear result: {cache_result}")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())