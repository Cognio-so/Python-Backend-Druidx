import os
import asyncio
import logging
import httpx
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Tuple
from urllib.parse import urlparse
from io import BytesIO
import base64

# Core imports
import openai
from openai import AsyncOpenAI
import numpy as np

from dotenv import load_dotenv
import json

load_dotenv()  # Load environment variables from .env file

# LangChain imports (following user requirements)
try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
    PDFPlumberLoader, Docx2txtLoader, BSHTMLLoader, TextLoader, UnstructuredURLLoader, JSONLoader
)
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    # MODIFIED: Import EnsembleRetriever for hybrid search
    from langchain.retrievers import EnsembleRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain packages not found. Some features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Import R2 Storage (following user requirements for integration)
try:
    from storage import CloudflareR2Storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logging.error("storage.py not found. Please ensure storage.py is available in the same directory.")

# Qdrant imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not found. Vector search will be unavailable.")

# --- START: ADD VISION-RELATED IMPORTS ---
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logging.warning("Anthropic package not found. Claude vision models will be unavailable.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google GenerativeAI SDK not found. Gemini vision models will be unavailable.")

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq Python SDK not found. Groq vision models will be unavailable.")

try:
    from PIL import Image
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    logging.warning("Pillow not found. Image processing will have limited fallbacks.")
# --- END: ADD VISION-RELATED IMPORTS ---


# BM25 for hybrid search (Enhanced with robust fallback like rag.py)
BM25_AVAILABLE = True
try:
    from langchain_community.retrievers import BM25Retriever
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
    logger.info("✅ BM25 package imported successfully")
except ImportError:
    # Implement our own simplified BM25 functionality (robust fallback)
    logger.warning("⚠️ Standard rank_bm25 import failed. Implementing custom BM25 solution...")
    
    # Custom BM25 implementation
    import numpy as np
    import uuid
    from langchain_core.retrievers import BaseRetriever
    from typing import Iterable, Callable
    from pydantic import Field, ConfigDict
    
    def default_preprocessing_func(text: str) -> List[str]:
        """Default preprocessing function that splits text on whitespace."""
        return text.lower().split()
    
    class BM25Okapi:
        """Simplified implementation of BM25Okapi when the rank_bm25 package is not available."""
        
        def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = []
            self.avgdl = 0
            self.N = 0
            
            if not self.corpus:
                return
                
            self.N = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.N
            
            # Calculate document frequencies
            for document in corpus:
                self.doc_len.append(len(document))
                freq = {}
                for word in document:
                    if word not in freq:
                        freq[word] = 0
                    freq[word] += 1
                self.doc_freqs.append(freq)
                
                # Update inverse document frequency
                for word, _ in freq.items():
                    if word not in self.idf:
                        self.idf[word] = 0
                    self.idf[word] += 1
            
            # Calculate inverse document frequency
            for word, freq in self.idf.items():
                self.idf[word] = np.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
        
        def get_scores(self, query):
            scores = [0] * self.N
            for q in query:
                if q not in self.idf:
                    continue
                q_idf = self.idf[q]
                for i, doc_freqs in enumerate(self.doc_freqs):
                    if q not in doc_freqs:
                        continue
                    doc_freq = doc_freqs[q]
                    doc_len = self.doc_len[i]
                    # BM25 scoring formula
                    numerator = q_idf * doc_freq * (self.k1 + 1)
                    denominator = doc_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += numerator / denominator
            return scores
        
        def get_top_n(self, query, documents, n=5):
            if not query or not documents or not self.N:
                return documents[:min(n, len(documents))]
            scores = self.get_scores(query)
            top_n = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:n]
            return [documents[i] for i in top_n]
    
    class SimpleBM25Retriever(BaseRetriever):
        """A simplified BM25 retriever implementation when rank_bm25 is not available."""
        vectorizer: Any = None
        docs: List[Document] = Field(default_factory=list, repr=False)
        k: int = 4
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )
        
        @classmethod
        def from_texts(
            cls,
            texts: Iterable[str],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[Iterable[str]] = None,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
        ) -> "SimpleBM25Retriever":
            """
            Create a SimpleBM25Retriever from a list of texts.
            """
            texts_list = list(texts)  # Convert iterable to list if needed
            texts_processed = [preprocess_func(t) for t in texts_list]
            bm25_params = bm25_params or {}
            # Create custom BM25Okapi vectorizer
            vectorizer = BM25Okapi(texts_processed, **bm25_params)
            
            # Create documents with metadata and ids
            documents = []
            metadatas = metadatas or ({} for _ in texts_list)
            if ids:
                documents = [
                    Document(page_content=t, metadata=m, id=i)
                    for t, m, i in zip(texts_list, metadatas, ids)
                ]
            else:
                documents = [
                    Document(page_content=t, metadata=m)
                    for t, m in zip(texts_list, metadatas)
                ]
            
            return cls(
                vectorizer=vectorizer,
                docs=documents,
                preprocess_func=preprocess_func,
                **kwargs
            )
        
        @classmethod
        def from_documents(
            cls,
            documents: Iterable[Document],
            *,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
        ) -> "SimpleBM25Retriever":
            """
            Create a SimpleBM25Retriever from a list of Documents.
            """
            documents_list = list(documents)  # Convert iterable to list if needed
            # Extract texts, metadatas, and ids from documents
            texts = []
            metadatas = []
            ids = []
            for doc in documents_list:
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
            
            return cls.from_texts(
                texts=texts,
                bm25_params=bm25_params,
                metadatas=metadatas,
                ids=ids,
                preprocess_func=preprocess_func,
                **kwargs,
            )
        
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            """Get documents relevant to a query."""
            processed_query = self.preprocess_func(query)
            if self.vectorizer and processed_query:
                return self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
            return self.docs[:min(self.k, len(self.docs))]
        
        # async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        #     """Asynchronously get documents relevant to a query."""
        #     # Async implementation just calls the sync version for simplicity
        #     return self._get_relevant_documents(query, run_manager=run_manager)
    
    # Replace the standard BM25Retriever with our custom implementation
    BM25Retriever = SimpleBM25Retriever
    BM25_AVAILABLE = True
    logger.info("✅ Custom BM25 implementation active - hybrid search enabled")


# Add these constants after the imports (around line 75, after the logger setup)
# Vector params for OpenAI's text-embedding-3-small (consistent with rag.py)
QDRANT_VECTOR_PARAMS = VectorParams(size=1536, distance=Distance.COSINE)
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"


@dataclass
class RAGConfig:
    """Configuration class for RAG system with R2 Storage integration"""
    
    # API Keys (kept for compatibility, but will be passed to LLMConfig)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    
    # Vector database settings
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_collection_name: str = "rag_documents"
    
    # LLM settings
    llm_model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 2000
    
    # OpenRouter settings
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    
    # Local Llama settings
    llama_model_path: str = field(default_factory=lambda: os.getenv("LLAMA_MODEL_PATH", ""))
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Retrieval settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    use_hybrid_search: bool = True
    
    # --- START: ADD VISION-RELATED CONFIG ---
    image_extensions: Tuple[str, ...] = field(default_factory=lambda: (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"))
    # --- END: ADD VISION-RELATED CONFIG ---

    # Storage settings (NEW - integrated from rag.py pattern)
    temp_processing_path: str = field(default_factory=lambda: os.getenv("TEMP_PROCESSING_PATH", "local_rag_data/temp_downloads"))
    enable_r2_storage: bool = True
    session_id: str = "default"  # For user document collections
    
    # Add current date and time to the query
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # Response settings - FIXED: Completely remove hallucination potential
    system_prompt: str = """You are a helpful AI assistant that provides accurate responses based STRICTLY on the provided context.

📌 **CRITICAL RESPONSE RULES - NO EXCEPTIONS:**
- **ALWAYS start with an emoji + quick headline**: 🏷️ **Your Headline**
- **ONLY use information from the provided context** - NEVER guess, assume, or use general knowledge
- **If context is insufficient, explicitly state:** "I don't have enough information in the provided context to answer this question."
- **For web search results, ALWAYS include the complete URL where information was found**

✅ **MANDATORY STYLE & FORMAT:**
- 🏷️ **Start with emoji + headline** (NEVER skip this)
- 📋 Use bullets or short paragraphs for clarity
- 💡 Emphasize main points
- 😊 Make it friendly and human
- 🤝 End with light follow-up when appropriate
- **ALWAYS include source references** when using provided context
- For document sources: "📄 **Source:** [Document Name]"
- For web sources: "🔗 **Source:** [URL]"

CRITICAL: NEVER provide information not found in the given context. Always cite your sources."""
    
    def to_llm_config(self) -> 'LLMConfig':
        """Convert RAGConfig to LLMConfig for LLM Manager"""
        if not LLM_MODEL_AVAILABLE:
            raise ImportError("llm_model module not available")
        
        return LLMConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            langsmith_api_key=self.langsmith_api_key,
            default_model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openrouter_base_url=self.openrouter_base_url,
            llama_model_path=self.llama_model_path,
            default_system_prompt=self.system_prompt
        )
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        return cls()

    def is_r2_url(self, url: str, r2_storage_client: Optional['CloudflareR2Storage'] = None) -> bool:
        """Check if a URL is an R2 storage URL"""
        if not r2_storage_client or not r2_storage_client.account_id:
            return False
        
        try:
            parsed_url = urlparse(url)
            return (
                f"{r2_storage_client.bucket_name}.{r2_storage_client.account_id}.r2.cloudflarestorage.com" in parsed_url.netloc
            )
        except Exception:
            return False

    def extract_r2_key_from_url(self, url: str) -> Optional[str]:
        """Extract R2 key from R2 URL"""
        try:
            parsed_url = urlparse(url)
            return parsed_url.path.lstrip('/')  # Remove leading /
        except Exception:
            return None


class DocumentProcessor:
    """Handles document loading and processing with R2 Storage integration"""
    
    # --- START: MODIFY __init__ TO ACCEPT RAGPipeline ---
    def __init__(self, config: RAGConfig, r2_storage_client: Optional['CloudflareR2Storage'], rag_pipeline: 'RAGPipeline'):
        self.config = config
        self.r2_storage_client = r2_storage_client
        self.rag_pipeline = rag_pipeline # Store reference to the main pipeline
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
        os.makedirs(config.temp_processing_path, exist_ok=True)
    # --- END: MODIFY __init__ ---

    async def load_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for file_path in file_paths:
            try:
                docs = await self._load_single_file(file_path)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue
        return documents
    
    async def _load_single_file(self, file_path: str) -> List[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PDFPlumberLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.json':
                loader = JSONLoader(file_path, jq_schema='.[*]', text_content=False)
            elif file_extension in ['.html', '.htm', '.xhtml']:
                loader = BSHTMLLoader(file_path)
            else:
                loader = TextLoader(file_path, autodetect_encoding=True)
            
            return await asyncio.to_thread(loader.load)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    async def _load_and_parse_from_memory(self, file_bytes: bytes, file_extension: str, source_url: str) -> List[Document]:
        """
        Loads and parses a file from an in-memory byte array.
        This replaces the need for temporary local files.
        """
        metadata = {"source": source_url, "title": os.path.basename(urlparse(source_url).path)}
        
        try:
            # --- START: ADD IMAGE HANDLING LOGIC ---
            if file_extension in self.config.image_extensions:
                if self.rag_pipeline.has_vision_capability:
                    logger.info(f"👁️ Processing image file: {source_url}")
                    image_content = await self.rag_pipeline._process_image_with_vision(file_bytes)
                    if image_content:
                         return [Document(page_content=image_content, metadata={**metadata, "file_type": "image"})]
                else:
                    logger.warning(f"⚠️ Skipping image file {source_url} as configured model lacks vision capabilities.")
                return []
            # --- END: ADD IMAGE HANDLING LOGIC ---

            if file_extension == '.pdf':
                try:
                    import pdfplumber
                    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                        content = "".join(page.extract_text(x_tolerance=1, y_tolerance=1) or "" for page in pdf.pages)
                    return [Document(page_content=content, metadata=metadata)]
                except ImportError:
                    logger.error("❌ pdfplumber package not found. Please install with `pip install pdfplumber` to process PDF files.")
                    return [Document(page_content="[Error: Could not process PDF content. The required 'pdfplumber' library is not installed on the server.]", metadata=metadata)]
            
            elif file_extension == '.docx':
                import docx2txt
                text = docx2txt.process(BytesIO(file_bytes))
                return [Document(page_content=text, metadata=metadata)]
                
            elif file_extension in ['.html', '.htm', '.xhtml']:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(file_bytes, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                return [Document(page_content=text, metadata=metadata)]

            elif file_extension == '.txt':
                text = file_bytes.decode('utf-8', errors='ignore')
                return [Document(page_content=text, metadata=metadata)]
                
            elif file_extension == '.json':
                try:
                    json_string = file_bytes.decode('utf-8', errors='ignore')
                    json_data = json.loads(json_string)
                    # Convert JSON to a nicely formatted string for the document content
                    content = json.dumps(json_data, indent=2, ensure_ascii=False)
                    logger.info(f"✅ Successfully parsed JSON content from {source_url}")
                    return [Document(page_content=content, metadata=metadata)]
                except json.JSONDecodeError as e_json:
                    logger.error(f"Failed to parse JSON from {source_url}: {e_json}. Treating as raw text.")
                    content = file_bytes.decode('utf-8', errors='ignore')
                    return [Document(page_content=content, metadata=metadata)]
                
            else:
                try:
                    text = file_bytes.decode('utf-8', errors='ignore')
                    logger.warning(f"Unsupported extension '{file_extension}' for memory loading. Treating as plain text.")
                    return [Document(page_content=text, metadata=metadata)]
                except Exception as e_decode:
                    logger.error(f"Could not decode file from {source_url} as text: {e_decode}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error parsing file from memory for source {source_url}: {e}", exc_info=True)
            return []

    async def load_documents_from_urls(self, urls: List[str]) -> List[Document]:
        documents = []
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            for url in urls:
                logger.info(f"Processing URL: {url}")
                file_bytes = None
                file_extension = os.path.splitext(urlparse(url).path)[1].lower()

                try:
                    is_our_r2_url = self.config.is_r2_url(url, self.r2_storage_client)
                    
                    if is_our_r2_url:
                        r2_key = self.config.extract_r2_key_from_url(url)
                        if r2_key and self.r2_storage_client:
                            logger.info(f"🔍 Detected R2 URL. Fetching content for key: {r2_key}")
                            file_bytes = await asyncio.to_thread(self.r2_storage_client.get_file_content_bytes, r2_key)
                    else:
                        logger.info(f"🌐 Detected external URL. Downloading content from: {url}")
                        response = await client.get(url)
                        response.raise_for_status()
                        file_bytes = response.content

                    if file_bytes:
                        docs = await self._load_and_parse_from_memory(file_bytes, file_extension, url)
                        if docs:
                            logger.info(f"✅ Successfully parsed {len(docs)} document(s) from {url}")
                            documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    continue
                
        return documents
    
    async def load_documents_from_r2_keys(self, r2_keys: List[str]) -> List[Document]:
        if not self.r2_storage_client:
            logger.error("R2 storage client not available")
            return []
        
        documents = []
        for r2_key in r2_keys:
            try:
                temp_filename = os.path.basename(r2_key)
                temp_path = os.path.join(self.config.temp_processing_path, temp_filename)
                
                success = self.r2_storage_client.download_file(r2_key, temp_path)
                if success:
                    docs = await self._load_single_file(temp_path)
                    for doc in docs:
                        doc.metadata['r2_key'] = r2_key
                        doc.metadata['source'] = f"r2://{r2_key}"
                    
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from R2 key {r2_key}")
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    logger.error(f"Failed to download R2 key: {r2_key}")
            except Exception as e:
                logger.error(f"Error loading R2 key {r2_key}: {e}")
                continue
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)


class VectorStoreManager:
    """Manages Qdrant vector store operations with R2 Storage integration"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.vector_store = None # This will represent the MAIN KB store
        self.embeddings = None
        
        if config.openai_api_key and LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(
                api_key=config.openai_api_key,
                model=config.embedding_model
            )
    
    async def initialize_collection(self, collection_name: Optional[str] = None):
        """Initialize a specific Qdrant collection"""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")
        
        target_collection = collection_name or self.config.qdrant_collection_name

        try:
            if not self.client:
                self.client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                    timeout=20.0
                )
            
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if target_collection not in collection_names:
                logger.info(f"Qdrant collection '{target_collection}' not found. Creating...")
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=target_collection,
                    vectors_config=QDRANT_VECTOR_PARAMS
                )
                logger.info(f"Created Qdrant collection: {target_collection}")
            else:
                logger.info(f"Using existing Qdrant collection: {target_collection}")

            # If this is the main KB collection, set the default vector_store attribute
            if target_collection == self.config.qdrant_collection_name:
                 if LANGCHAIN_AVAILABLE and self.embeddings:
                    self.vector_store = QdrantVectorStore(
                        client=self.client,
                        collection_name=target_collection,
                        embedding=self.embeddings,
                        content_payload_key=CONTENT_PAYLOAD_KEY,
                        metadata_payload_key=METADATA_PAYLOAD_KEY
                    )
                    logger.info(f"Initialized default vector store for main KB: {target_collection}")

        except Exception as e:
            logger.error(f"Error initializing Qdrant collection '{target_collection}': {e}")
            raise
    
    async def add_documents(self, documents: List[Document], collection_name: Optional[str] = None) -> List[str]:
        """Add documents to a specific vector store collection"""
        target_collection = collection_name or self.config.qdrant_collection_name
        
        if not self.client or not self.embeddings:
            raise ValueError("Vector store manager not fully initialized")

        try:
            # Ensure the target collection exists
            await self.initialize_collection(target_collection)

            # Create a QdrantVectorStore instance specifically for the target collection
            store_for_add = QdrantVectorStore(
                client=self.client,
                collection_name=target_collection,
                embedding=self.embeddings,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )

            doc_ids = await store_for_add.aadd_documents(documents)
            logger.info(f"Added {len(documents)} documents to collection '{target_collection}'")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents to collection '{target_collection}': {e}")
            raise
    
    def get_retriever(self, k: int = None, collection_name: Optional[str] = None) -> BaseRetriever:
        """Get a retriever for a specific vector store collection"""
        target_collection = collection_name or self.config.qdrant_collection_name
        
        if not self.client or not self.embeddings:
            raise ValueError("Vector store manager not fully initialized")

        # Use the default vector_store if the target is the main KB, otherwise create a new instance
        if target_collection == self.config.qdrant_collection_name and self.vector_store:
            store_to_use = self.vector_store
        else:
            store_to_use = QdrantVectorStore(
                client=self.client,
                collection_name=target_collection,
                embedding=self.embeddings,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )
        
        search_k = k or self.config.retrieval_k
        return store_to_use.as_retriever(search_kwargs={"k": search_k})
    
    async def clear_collection(self, collection_name: Optional[str] = None):
        """Clear all documents from a specific collection"""
        target_collection = collection_name or self.config.qdrant_collection_name

        if not self.client:
            raise ValueError("Qdrant client not initialized")
        
        try:
            logger.info(f"Attempting to clear vector store collection '{target_collection}'")
            await asyncio.to_thread(
                self.client.delete_collection, 
                collection_name=target_collection
            )
            # Recreate the collection after clearing it
            await self.initialize_collection(target_collection)
            logger.info(f"Successfully cleared and re-initialized collection '{target_collection}'")
        except Exception as e:
            # It's okay if the collection didn't exist
            if "not found" in str(e).lower() or "404" in str(e):
                logger.warning(f"Collection '{target_collection}' did not exist, so no need to clear. Re-initializing.")
                await self.initialize_collection(target_collection)
            else:
                logger.error(f"Error clearing collection '{target_collection}': {e}")
                raise
    
    async def get_existing_documents_count(self, collection_name: Optional[str] = None) -> int:
        """Get count of existing documents in a specific collection"""
        target_collection = collection_name or self.config.qdrant_collection_name
        if not self.client:
            return 0
        
        try:
            collection_info = await asyncio.to_thread(self.client.get_collection, target_collection)
            return collection_info.points_count
        except Exception as e:
            # If collection doesn't exist, count is 0
            if "not found" in str(e).lower() or "404" in str(e):
                return 0
            logger.error(f"Error getting document count for '{target_collection}': {e}")
            return 0
    
    async def cleanup(self):
        """Cleanup Qdrant client"""
        if hasattr(self, 'client') and self.client:
            try:
                # QdrantClient doesn't have async close, but we can reset it
                self.client = None
                logger.info("✅ Qdrant client cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up Qdrant client: {e}")

# REMOVED: The old HybridRetriever class is no longer needed.
# It will be replaced by LangChain's EnsembleRetriever.

class RAGPipeline:
    """Enhanced RAG Pipeline with improved similarity filtering and cosine similarity support"""
    
    def __init__(self, config: RAGConfig, r2_storage_client: Optional['CloudflareR2Storage'] = None):
        self.config = config
        self.r2_storage_client = r2_storage_client or (CloudflareR2Storage() if STORAGE_AVAILABLE else None)
        
        # --- START: MODIFY DocumentProcessor INITIALIZATION ---
        # Pass `self` (the RAGPipeline instance) to the DocumentProcessor
        self.document_processor = DocumentProcessor(config, self.r2_storage_client, self)
        # --- END: MODIFY DocumentProcessor INITIALIZATION ---

        self.vector_store_manager = VectorStoreManager(config)
        
        # Initialize centralized LLM Manager
        if not LLM_MODEL_AVAILABLE:
            raise ImportError("llm_model module not available. Please ensure llm_model.py is in the same directory.")
        
        llm_config = config.to_llm_config()
        self.llm_manager = LLMManager(llm_config)
        
        # --- START: INITIALIZE PROVIDER CLIENTS FOR VISION ---
        # These clients are used *only* for vision, bypassing the LLMManager
        # which doesn't support multimodal inputs in its current form.
        timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
        self.vision_openai_client = AsyncOpenAI(api_key=self.config.openai_api_key, timeout=timeout_config)
        self.vision_anthropic_client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key) if CLAUDE_AVAILABLE and self.config.anthropic_api_key else None
        if GEMINI_AVAILABLE and self.config.google_api_key:
            genai.configure(api_key=self.config.google_api_key)
            self.vision_gemini_client = genai
        else:
            self.vision_gemini_client = None
        self.vision_groq_client = AsyncGroq(api_key=self.config.groq_api_key) if GROQ_AVAILABLE and self.config.groq_api_key else None
        # --- END: INITIALIZE PROVIDER CLIENTS FOR VISION ---

        # --- START: ADD VISION CAPABILITY CHECK ---
        self.has_vision_capability = self.config.llm_model.lower() in [
            "gpt-4o", "gpt-4o-mini", "gpt-4-vision", "gpt-4-turbo",
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-1.5-flash-latest",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3.5-sonnet-20240620",
            "llava-v1.5-7b"
        ] or "vision" in self.config.llm_model.lower()

        if self.has_vision_capability:
            logger.info(f"✅ Vision capabilities ENABLED for model: {self.config.llm_model}")
        else:
            logger.warning(f"⚠️ Vision capabilities DISABLED for model: {self.config.llm_model}. Image files will be skipped.")
        # --- END: ADD VISION CAPABILITY CHECK ---

        # MODIFIED: Use a single `retriever` attribute for the ensemble retriever
        self.retriever = None
        self.documents = []
        
        # Setup LangSmith if available
        if LANGSMITH_AVAILABLE and config.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
            os.environ["LANGSMITH_TRACING"] = "true"
        
        logger.info(f"✅ RAG Pipeline initialized with {'R2 storage' if self.r2_storage_client else 'no storage'}")
    
    # --- START: ADD VISION PROCESSING METHOD ---
    async def _process_image_with_vision(self, image_data: bytes) -> Optional[str]:
        """Process an image using a vision-capable model, returning a text description."""
        if not self.has_vision_capability:
            return None

        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            model_name = self.config.llm_model.lower()
            prompt_text = "Describe the content of this image in detail, including any text, objects, people, and the overall scene."

            # OpenAI Models
            if "gpt-" in model_name:
                vision_model = "gpt-4o" # Use gpt-4o as the vision workhorse
                logger.info(f"Using OpenAI model '{vision_model}' for vision analysis.")
                response = await self.vision_openai_client.chat.completions.create(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    max_tokens=1024
                )
                return f"Image Content (Analyzed by {vision_model}):\n{response.choices[0].message.content}"

            # Gemini Models
            elif "gemini" in model_name and self.vision_gemini_client:
                vision_model = "gemini-1.5-flash" if "flash" in model_name else "gemini-1.5-pro"
                logger.info(f"Using Gemini model '{vision_model}' for vision analysis.")
                model_instance = self.vision_gemini_client.GenerativeModel(vision_model)
                response = await model_instance.generate_content_async([prompt_text, {"mime_type": "image/jpeg", "data": base64.b64decode(base64_image)}])
                return f"Image Content (Analyzed by {vision_model}):\n{response.text}"

            # Claude Models
            elif "claude" in model_name and self.vision_anthropic_client:
                # Map friendly names to specific model IDs
                vision_model_map = {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
                    "claude-3-sonnet": "claude-3-sonnet-20240229",
                    "claude-3-haiku": "claude-3-haiku-20240307",
                }
                vision_model = next((v for k, v in vision_model_map.items() if k in model_name), "claude-3-haiku-20240307")
                logger.info(f"Using Anthropic model '{vision_model}' for vision analysis.")
                response = await self.vision_anthropic_client.messages.create(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                        ]
                    }],
                    max_tokens=1024
                )
                return f"Image Content (Analyzed by {vision_model}):\n{response.content[0].text}"

        except Exception as e:
            logger.error(f"Error during vision processing with model {model_name}: {e}")
            if IMAGE_PROCESSING_AVAILABLE:
                try:
                    img = Image.open(BytesIO(image_data))
                    return f"[Image file: {img.width}x{img.height} {img.format}. Vision analysis failed: {e}]"
                except Exception as img_e:
                    logger.error(f"Pillow fallback failed: {img_e}")

        return "[Image file could not be processed.]"
    # --- END: ADD VISION PROCESSING METHOD ---

    async def initialize(self):
        """Initialize the RAG pipeline"""
        try:
            await self.vector_store_manager.initialize_collection()
            
            # MODIFIED: Initialize the retriever with a fallback mechanism.
            # This sets up the hybrid retriever structure, even if the BM25 part is initially empty.
            vector_retriever = self.vector_store_manager.get_retriever()
            
            # Check for existing documents to log information
            existing_count = await self.vector_store_manager.get_existing_documents_count()
            if existing_count > 0:
                logger.info(f"Found {existing_count} existing documents in knowledge base.")
            else:
                logger.info("No existing documents found in knowledge base.")

            # Always try to initialize the hybrid retriever. It will be updated when documents are added.
            self._update_retriever()

            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise
    
    async def add_documents_from_files(self, file_paths: List[str]) -> List[str]:
        """Add documents from files to the knowledge base"""
        try:
            # Load and process documents
            raw_documents = await self.document_processor.load_documents_from_files(file_paths)
            split_documents = self.document_processor.split_documents(raw_documents)
            
            # Add to vector store
            doc_ids = await self.vector_store_manager.add_documents(split_documents)
            
            # Update documents list for hybrid retrieval
            self.documents.extend(split_documents)
            # MODIFIED: Update the ensemble retriever
            self._update_retriever()
            
            logger.info(f"Successfully added {len(split_documents)} document chunks from {len(file_paths)} files")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents from files: {str(e)}")
            raise
    
    async def add_documents_from_urls(self, urls: List[str]) -> List[str]:
        """Add documents from URLs to the knowledge base"""
        try:
            # Load and process documents
            raw_documents = await self.document_processor.load_documents_from_urls(urls)
            if not raw_documents:
                logger.warning(f"No documents were successfully processed from the provided URLs: {urls}")
                return []
            split_documents = self.document_processor.split_documents(raw_documents)
            
            # Add to vector store
            doc_ids = await self.vector_store_manager.add_documents(split_documents)
            
            # Update documents list for hybrid retrieval
            self.documents.extend(split_documents)
            # MODIFIED: Update the ensemble retriever
            self._update_retriever()
            
            logger.info(f"Successfully added {len(split_documents)} document chunks from {len(urls)} URLs")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents from URLs: {str(e)}")
            raise
    
    async def add_documents_from_r2_keys(self, r2_keys: List[str]) -> List[str]:
        """Add documents from R2 storage keys to the knowledge base (NEW METHOD)"""
        try:
            # Load and process documents from R2
            raw_documents = await self.document_processor.load_documents_from_r2_keys(r2_keys)
            split_documents = self.document_processor.split_documents(raw_documents)
            
            # Add to vector store
            doc_ids = await self.vector_store_manager.add_documents(split_documents)
            
            # Update documents list for hybrid retrieval
            self.documents.extend(split_documents)
            # MODIFIED: Update the ensemble retriever
            self._update_retriever()
            
            logger.info(f"Successfully added {len(split_documents)} document chunks from {len(r2_keys)} R2 keys")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents from R2 keys: {str(e)}")
            raise
    
    async def add_text_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add text documents directly to the knowledge base"""
        try:
            # Create documents from texts
            documents = []
            metadatas = metadatas or [{}] * len(texts)
            
            for text, metadata in zip(texts, metadatas):
                documents.append(Document(page_content=text, metadata=metadata))
            
            # Split documents
            split_documents = self.document_processor.split_documents(documents)
            
            # Add to vector store
            doc_ids = await self.vector_store_manager.add_documents(split_documents)
            
            # Update documents list for hybrid retrieval
            self.documents.extend(split_documents)
            # MODIFIED: Update the ensemble retriever
            self._update_retriever()
            
            logger.info(f"Successfully added {len(split_documents)} document chunks from {len(texts)} texts")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding text documents: {str(e)}")
            raise
    
    async def add_documents_from_local_paths(self, local_paths: List[str]) -> Dict[str, Any]:

        """
        FIXED: Implemented this method.
        Loads documents from local file paths, processes them, and adds them to the knowledge base.
        """
        logger.info(f"RAG Pipeline: Adding {len(local_paths)} documents from local paths.")
        try:
            # Step 1: Load documents from the local file paths using the document processor.
            raw_documents = await self.document_processor.load_documents_from_files(local_paths)
            if not raw_documents:
                logger.warning("No documents were loaded from the provided local paths.")
                return {"success": True, "message": "No documents loaded.", "doc_ids": []}

            # Step 2: Split the loaded documents into manageable chunks.
            split_documents = self.document_processor.split_documents(raw_documents)
            logger.info(f"Split {len(raw_documents)} raw documents into {len(split_documents)} chunks.")

            # Step 3: Add the document chunks to the vector store.
            doc_ids = await self.vector_store_manager.add_documents(split_documents)
            
            # Step 4: Update the in-memory document list for hybrid search.
            self.documents.extend(split_documents)
        
            # Step 5: Update the retriever to include the new documents.
            self._update_retriever()
            
            logger.info(f"Successfully added {len(split_documents)} document chunks from {len(local_paths)} files.")
            
            return {"success": True, "message": f"Successfully indexed {len(split_documents)} chunks.","doc_ids": doc_ids}
        except Exception as e:
            logger.error(f"Error adding documents from local paths: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # MODIFIED: Renamed from _update_hybrid_retriever and implemented EnsembleRetriever
    def _update_retriever(self):
        """
        Update the retriever with the current set of documents.
        Uses a hybrid ensemble approach combining BM25 and Qdrant vector search.
        """
        try:
            vector_retriever = self.vector_store_manager.get_retriever()

            # Fallback to just the vector retriever if BM25 is not available or there are no documents for it
            if not BM25_AVAILABLE or not self.documents:
                self.retriever = vector_retriever
                logger.info("Updated retriever to use QdrantVectorStore only (BM25 not available or no in-memory documents).")
                return

            # Initialize BM25 retriever with the in-memory documents
            bm25_retriever = BM25Retriever.from_documents(
                self.documents, 
                k=self.config.retrieval_k * 2  # Get more candidates for reranking
            )
            
            # Create the ensemble retriever with specified weights
            self.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7]  # Weight for BM25 and Vector Search respectively
            )
            logger.info(f"✅ Updated hybrid ensemble retriever with {len(self.documents)} documents.")
        
        except Exception as e:
            logger.error(f"Error updating ensemble retriever: {str(e)}. Falling back to vector store retriever.")
            # Ensure there's always a retriever available
            if self.vector_store_manager:
                self.retriever = self.vector_store_manager.get_retriever()
            else:
                self.retriever = None
    
    async def get_relevant_documents(self, query: str, k: int = 10) -> List[Document]:
        """Get relevant documents from the RAG system using the configured retriever."""
        try:
            if not self.retriever:
                logger.warning("Retriever not available, returning empty list")
                return []
            
            # The 'k' parameter is now configured on the individual retrievers (BM25, Qdrant).
            # The EnsembleRetriever handles combining the results.
            relevant_docs = await self.retriever.ainvoke(query)
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return []
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Enhanced document formatting with proper categorization and emoji headers"""
        if not docs:
            return "No relevant documents found."
        
        # Categorize documents by type
        web_docs = []
        user_docs = []
        rag_docs = []
        
        for doc in docs:
            source_type = doc.metadata.get("source_type", "")
            source = doc.metadata.get("source", "").lower()
            
            if source_type == "web_search" or "web search" in source:
                web_docs.append(doc)
            elif "user" in source or source_type == "user_provided":
                user_docs.append(doc)
            else:
                rag_docs.append(doc)
        
        formatted_sections = []
        
        # Format Web Search Results First (if any)
        if web_docs:
            formatted_sections.append("## 🌐 **WEB SEARCH RESULTS**")
            for i, doc in enumerate(web_docs, 1):
                title = doc.metadata.get('title', f'Web Result {i}')
                url = doc.metadata.get('url', '')
                content = doc.page_content.strip()
                
                # Create clickable link
                header = f"### 📰 **{title}**"
                if url:
                    header += f"\n🔗 **Source:** {url}"
                
                formatted_sections.append(f"{header}\n\n{content}")
        
        # Format User Documents (if any)
        if user_docs:
            if web_docs:  # Add separator if we have web docs
                formatted_sections.append("---")
            formatted_sections.append("## 📄 **USER DOCUMENTS**")
            for i, doc in enumerate(user_docs, 1):
                source = doc.metadata.get('source', f'User Document {i}')
                content = doc.page_content.strip()
                
                header = f"### 📄 **{source}**"
                formatted_sections.append(f"{header}\n\n{content}")
        
        # Format Knowledge Base Documents (if any)
        if rag_docs:
            if web_docs or user_docs:  # Add separator if we have other docs
                formatted_sections.append("---")
            formatted_sections.append("## 📚 **KNOWLEDGE BASE**")
            for i, doc in enumerate(rag_docs, 1):
                source = doc.metadata.get('source', f'Knowledge Base {i}')
                content = doc.page_content.strip()
                
                header = f"### 📚 **{source}**"
                formatted_sections.append(f"{header}\n\n{content}")
        
        return "\n\n".join(formatted_sections)
    
    def _create_followup_aware_prompt(self, system_prompt_override: Optional[str] = None, is_follow_up: bool = False, referring_entity: Optional[str] = None) -> ChatPromptTemplate:
        """Create a prompt template that's aware of follow-up context"""
        
        system_message = system_prompt_override or """You are a helpful AI assistant that provides accurate responses based STRICTLY on the provided context.

📌 **CRITICAL RESPONSE RULES - NO EXCEPTIONS:**
- **ALWAYS start with an emoji + quick headline**: 🏷️ **Your Headline**
- **ONLY use information from the provided context** - NEVER guess, assume, or use general knowledge
- **If context is insufficient, explicitly state:** "I don't have enough information in the provided context to answer this question."
- **For web search results, ALWAYS include the complete URL where information was found**

✅ **MANDATORY STYLE & FORMAT:**
- 🏷️ **Start with emoji + headline** (NEVER skip this)
- 📋 Use bullets or short paragraphs for clarity
- 💡 Emphasize main points
- 😊 Make it friendly and human
- 🤝 End with light follow-up when appropriate
- **ALWAYS include source references** when using provided context
- For document sources: "📄 **Source:** [Document Name]"
- For web sources: "🔗 **Source:** [URL]"

CRITICAL: NEVER provide information not found in the given context. Always cite your sources."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", """Context Information:
{context}

Current Time: {current_time}

User Question: {question}

CRITICAL INSTRUCTION: You MUST start your response with emoji + headline format (🏷️ **Your Headline**). Use ONLY the provided context with proper source citations. If the context doesn't contain enough information to answer the question, clearly state that instead of guessing.""")
        ])
    
    def _create_document_retriever_runnable(self):
        """Create a runnable for document retrieval"""
        async def retrieve_docs(input_dict):
            question = input_dict.get("question", "")
            if self.retriever:
                docs = await self.retriever.ainvoke(question)
                return self._format_documents(docs)
            return "No documents available."
        
        return RunnableLambda(retrieve_docs)
    
    def _get_llm_runnable(self, model: str = None, system_prompt_override: Optional[str] = None):
        """Get LLM as a runnable for LCEL chains using centralized LLM Manager"""
        selected_model = model or self.config.llm_model
        
        async def process_llm_input(input_dict):
            """Process input through the centralized LLM manager"""
            try:
                # Handle different input formats
                if isinstance(input_dict, str):
                    formatted_prompt = input_dict
                elif isinstance(input_dict, dict):
                    formatted_prompt = input_dict.get("text", str(input_dict))
                else:
                    formatted_prompt = str(input_dict)
                
                # Convert to messages format
                messages = [
                    {"role": "user", "content": formatted_prompt}
                ]
                
                # Generate response using centralized LLM manager
                response = await self.llm_manager.generate_response(
                    messages=messages,
                    model=selected_model,
                    stream=False,
                    system_prompt=system_prompt_override
                )
                
                return response
            except Exception as e:
                logger.error(f"Error in LLM runnable: {e}")
                return f"Error processing request: {str(e)}"
        
        return RunnableLambda(process_llm_input)
    
    def _get_current_time_runnable(self):
        """Get current time as a runnable"""
        return RunnableLambda(lambda x: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @traceable(name="rag_query") if LANGSMITH_AVAILABLE else lambda f: f
    async def query(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None, 
                   model: str = None, pre_retrieved_docs: Optional[List[Document]] = None,
                   system_prompt_override: Optional[str] = None) -> str:
        try:
            # Use pre-retrieved documents if provided, otherwise retrieve
            if pre_retrieved_docs is not None:
                relevant_docs = pre_retrieved_docs
                logger.info(f"📚 Using {len(relevant_docs)} pre-retrieved documents")
            else:
                # Check if we have documents and retriever
                if not self.retriever:
                    # No RAG documents available - use general knowledge with proper formatting
                    logger.info("🔍 No RAG retriever available, using general knowledge with formatting enforcement")
                    
                    # Create messages with strong format enforcement
                    messages = [
                        {"role": "system", "content": system_prompt_override or f"""{self.config.system_prompt}

CRITICAL FORMAT ENFORCEMENT: You MUST start your response with emoji + headline format: "🏷️ **Your Headline**". This is mandatory for ALL responses."""},
                        {"role": "user", "content": f"""Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

User Question: {question}

CRITICAL INSTRUCTION: You MUST start your response with the emoji + headline format (🏷️ **Your Headline**). This is mandatory even when providing general knowledge. Since no relevant documents were found, clearly state: "I don't have enough information in the provided context to answer this question." """}
                    ]
                    
                    # Generate response using centralized LLM manager with format enforcement
                    response = await self.llm_manager.generate_response(
                        messages=messages,
                        model=model or self.config.llm_model,
                        stream=False
                    )
                    
                    return response
                
                # Retrieve relevant documents
                relevant_docs = await self.retriever.ainvoke(question,limit=self.config.retrieval_k)

            if not relevant_docs:
                # No relevant documents found - clearly state this
                logger.info("🔍 No relevant documents found")
                
                insufficient_response = f"""🏷️ **Information Not Available**

I don't have enough information in the provided context to answer your question about "{question}".

The knowledge base doesn't contain relevant documents that can help me provide an accurate answer.

🤝 Could you try rephrasing your question or providing more specific details?"""
                
                return insufficient_response
            
            # We have relevant documents - use full RAG pipeline
            logger.info(f"📚 Found {len(relevant_docs)} relevant documents for RAG response")
            
            # Create prompt template
            prompt_template = self._create_followup_aware_prompt(system_prompt_override)
            
            rag_chain = (
                RunnableParallel({
                    "context": RunnableLambda(lambda x: self._format_documents(relevant_docs)),
                    "question": RunnableLambda(lambda x: x["question"]),
                    "current_time": self._get_current_time_runnable()
                })
                | prompt_template
                | self._get_llm_runnable(model, system_prompt_override)
                | StrOutputParser()
            )
            
            # Execute the chain
            response = await rag_chain.ainvoke({"question": question})
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return f"🚨 **Error Processing Request**\n\nI encountered an error while processing your question: {str(e)}"
    
    @traceable(name="rag_query_stream") if LANGSMITH_AVAILABLE else lambda f: f
    async def query_stream(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None,
                          model: str = None, pre_retrieved_docs: Optional[List[Document]] = None,
                          system_prompt_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Query the RAG system using LCEL chain and return a streaming response"""
        try:
            # Use pre-retrieved documents if provided
            if pre_retrieved_docs is not None:
                relevant_docs = pre_retrieved_docs
                logger.info(f"📚 Using {len(relevant_docs)} pre-retrieved documents for streaming")
            else:
                # Check if we have documents and retriever
                if not self.retriever:

                    logger.info("🔍 No RAG retriever available, using general knowledge with formatting enforcement")
                    
                    messages = [
                        {"role": "system", "content": system_prompt_override or f"""{self.config.system_prompt}

    CRITICAL FORMAT ENFORCEMENT: You MUST start your response with emoji + headline format: "🏷️ **Your Headline**". This is mandatory for ALL responses."""},
                        {"role": "user", "content": f"""Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    User Question: {question}

    CRITICAL INSTRUCTION: You MUST start your response with the emoji + headline format (🏷️ **Your Headline**). This is mandatory even when providing general knowledge. Use *General Insight:* when providing information from your training data."""}
                    ]
                    
                    if chat_history:
                        current_msg = messages.pop()
                        messages.extend(chat_history[-5:])
                        messages.append(current_msg)

                    # This ensures every message sent to the LLM is a valid dictionary.
                    sanitized_messages = []
                    for i, msg in enumerate(messages):
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            sanitized_messages.append(msg)
                        elif isinstance(msg, tuple) and len(msg) == 2:
                            logger.warning(f"Corrected a malformed tuple-based message at index {i}.")
                            sanitized_messages.append({"role": str(msg[0]), "content": str(msg[1])})
                        else:
                            logger.warning(f"Skipping a malformed message at index {i}: {type(msg)}")

                    response_stream = await self.llm_manager.generate_response(
                        messages=sanitized_messages, # Use the sanitized list
                        model=model or self.config.llm_model,
                        stream=True
                    )
                    
                    async for chunk in response_stream:
                        yield chunk
                    return
                
                # Retrieve relevant documents
                relevant_docs = await self.retriever.ainvoke(question)

            if not relevant_docs:
                logger.info("🔍 No relevant documents found, using general knowledge with formatting enforcement")
                
                messages = [
                    {"role": "system", "content": system_prompt_override or f"""{self.config.system_prompt}

CRITICAL FORMAT ENFORCEMENT: You MUST start your response with emoji + headline format: "🏷️ **Your Headline**". This is mandatory for ALL responses."""},
                    {"role": "user", "content": f"""Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

User Question: {question}

CRITICAL INSTRUCTION: You MUST start your response with the emoji + headline format (🏷️ **Your Headline**). No relevant documents were found, so provide the best answer you can with general knowledge. Use *General Insight:* when providing information from your training data."""}
                ]
                
                # Add chat history if provided
                if chat_history:
                    current_msg = messages.pop()
                    messages.extend(chat_history[-5:])
                    messages.append(current_msg)
                
                response_stream = await self.llm_manager.generate_response(
                    messages=messages,
                    model=model or self.config.llm_model,
                    stream=True
                )
                
                async for chunk in response_stream:
                    yield chunk
                return
            
            # We have relevant documents - use full RAG pipeline
            logger.info(f"📚 Found {len(relevant_docs)} relevant documents for RAG response")
            
            # Create prompt template
            prompt_template = self._create_followup_aware_prompt(system_prompt_override)
            
            # Build context
            context = self._format_documents(relevant_docs)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format the final prompt
            formatted_prompt = await prompt_template.aformat_prompt(
                context=context,
                question=question,
                current_time=current_time
            )
            
            # Convert to messages format for LLM
            messages = []
            
            # Add system message if it exists
            if hasattr(formatted_prompt, 'messages') and formatted_prompt.messages:
                for msg in formatted_prompt.messages:
                    if hasattr(msg, 'content'):
                        if hasattr(msg, 'type'):
                            if msg.type == 'system':
                                messages.append({"role": "system", "content": msg.content})
                            elif msg.type == 'human':
                                messages.append({"role": "user", "content": msg.content})
                        else:
                            messages.append({"role": "user", "content": msg.content})
            else:
                # Fallback formatting with strong format enforcement
                messages.append({"role": "system", "content": system_prompt_override or f"""{self.config.system_prompt}

CRITICAL FORMAT ENFORCEMENT: You MUST start your response with emoji + headline format: "🏷️ **Your Headline**". This is mandatory for ALL responses."""})
                user_message = f"""Context Information:
{context}

Current Time: {current_time}

User Question: {question}

CRITICAL INSTRUCTION: You MUST start your response with emoji + headline format (🏷️ **Your Headline**). Use the provided context with proper source citations, and format according to the style guide. Include source references for all information used."""
                messages.append({"role": "user", "content": user_message})
            
            # Add chat history if provided
            if chat_history:
                current_msg = messages.pop()
                messages.extend(chat_history[-5:])
                messages.append(current_msg)
            
            # Sanitize before the final call
            sanitized_messages = []
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    sanitized_messages.append(msg)
                elif isinstance(msg, tuple) and len(msg) == 2:
                    logger.warning(f"Corrected a malformed tuple-based message at index {i}.")
                    sanitized_messages.append({"role": str(msg[0]), "content": str(msg[1])})
                else:
                    logger.warning(f"Skipping a malformed message at index {i}: {type(msg)}")

            response_stream = await self.llm_manager.generate_response(
                messages=sanitized_messages, # Use the sanitized list
                model=model or self.config.llm_model,
                stream=True
            )
            
            async for chunk in response_stream:
                yield chunk
                    
        except Exception as e:
            logger.error(f"Error in RAG query stream: {str(e)}")
            yield f"🚨 **Error Processing Request**\n\nI encountered an error while processing your question: {str(e)}"


    async def cleanup(self):
        """Cleanup RAG pipeline resources"""
        try:
            # Cleanup vector store manager
            if hasattr(self, 'vector_store_manager') and self.vector_store_manager:
                await self.vector_store_manager.cleanup()
            
            # Cleanup LLM manager
            if hasattr(self, 'llm_manager') and self.llm_manager:
                await self.llm_manager.cleanup()
            
            logger.info("✅ RAG pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during RAG pipeline cleanup: {e}")