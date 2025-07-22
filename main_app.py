from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import shutil
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
from io import BytesIO
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
import logging
import signal
import sys
import re
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the ChatbotAgent and related components
try:
    from agent import ChatbotAgent, ChatbotConfig
    AGENT_AVAILABLE = True
    logger.info("✅ ChatbotAgent imported successfully")
except ImportError as e:
    AGENT_AVAILABLE = False
    logger.error(f"❌ Failed to import ChatbotAgent: {e}. Please ensure agent.py is in the same directory.")

# Import storage for direct operations
try:
    from storage import CloudflareR2Storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.error("❌ Storage module not available. Please ensure storage.py is in the same directory.")

try:
    from llm_model import LLMManager, LLMConfig
    from qdrant_client import QdrantClient
    from rag_code import RAGConfig # To get Qdrant URL
    LLM_MODEL_AVAILABLE = True
    QDRANT_AVAILABLE = True
except ImportError as e:
    LLM_MODEL_AVAILABLE = False
    QDRANT_AVAILABLE = False
    logger.error(f"❌ Failed to import core clients (LLMManager, QdrantClient): {e}")

# Global variables
active_agents: Dict[str, ChatbotAgent] = {}
agents_lock = asyncio.Lock()
r2_storage: Optional[CloudflareR2Storage] = None
global_llm_manager: Optional[LLMManager] = None
global_qdrant_client: Optional[QdrantClient] = None

# Initialize paths
LOCAL_DATA_BASE_PATH = os.getenv("LOCAL_DATA_PATH", "local_rag_data")
TEMP_DOWNLOAD_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "temp_downloads")
os.makedirs(TEMP_DOWNLOAD_PATH, exist_ok=True)

# --- Lifespan and Cleanup Functions ---

async def cleanup_r2_expired_files():
    """Periodic task to clean up expired R2 files"""
    global r2_storage
    logger.info("🧹 Running scheduled cleanup of expired R2 files...")
    if r2_storage and not r2_storage.use_local_fallback:
        try:
            deleted_count = await asyncio.to_thread(r2_storage.check_and_delete_expired_files)
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} expired files")
            else:
                logger.info("🧹 No expired files to clean up.")
        except Exception as e:
            logger.error(f"❌ Error during scheduled R2 cleanup: {e}")
    else:
        logger.warning("⚠️ R2 storage not available or in fallback mode; skipping cleanup task.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle, including the single R2 storage client."""
    global r2_storage, global_llm_manager, global_qdrant_client
    logger.info("🚀 Starting FastAPI application with ChatbotAgent middleware...")
    
    if STORAGE_AVAILABLE:
        try:
            # Create a single, shared instance of the storage client.
            r2_storage = CloudflareR2Storage()
            if r2_storage.use_local_fallback:
                 logger.warning("✅ R2 storage initialized in local fallback mode.")
            else:
                 logger.info("✅ R2 storage initialized successfully.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to initialize R2 storage on startup: {e}")
            r2_storage = None # Ensure it's None on failure

    if LLM_MODEL_AVAILABLE:
        try:
            global_llm_manager = LLMManager(LLMConfig.from_env())
            logger.info("✅ Global LLM Manager initialized successfully.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to initialize Global LLM Manager: {e}")
    
    if QDRANT_AVAILABLE:
        try:
            rag_config = RAGConfig.from_env()
            global_qdrant_client = QdrantClient(
                url=rag_config.qdrant_url,
                api_key=rag_config.qdrant_api_key,
                timeout=30.0
            )
            # Perform a quick operation to "warm up" the connection
            await asyncio.to_thread(global_qdrant_client.get_collections)
            logger.info("✅ Global Qdrant Client initialized and connection warmed up.")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to initialize Global Qdrant Client: {e}")
    
    scheduler = AsyncIOScheduler()
    scheduler.add_job(cleanup_r2_expired_files, 'interval', hours=6)
    scheduler.start()
    logger.info("⏰ Scheduler started: R2 cleanup will run every 6 hours")
    
    yield
    
    logger.info("🛑 Shutting down application...")
    scheduler.shutdown(wait=False)
    logger.info("✅ Scheduler stopped")

    if global_llm_manager:
        await global_llm_manager.cleanup()
        logger.info("✅ Global LLM Manager cleaned up.")

    if global_qdrant_client:
        # Qdrant client might not have an async close, depends on version.
        # Setting to None is sufficient if no explicit close method.
        global_qdrant_client = None
        logger.info("✅ Global Qdrant Client closed.")
    
    async with agents_lock:
        cleanup_tasks = [agent.cleanup() for agent in active_agents.values()]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        active_agents.clear()
        logger.info("✅ All agents cleaned up")

def setup_signal_handlers():
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        # Let the lifespan manager handle cleanup
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Enhanced Chatbot API with Agent Middleware",
    description="A compatible API layer for ChatbotAgent, supporting schemas from both main_rag.py and the new agent architecture.",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://my-gpt-frontend.vercel.app", "https://www.druidx.co", "http://localhost:3000", "https://www.mygpt.work", "https://www.EMSA.co"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

setup_signal_handlers()

# --- Pydantic Models (Merged from main_rag.py and main_app.py) ---

class BaseAgentRequest(BaseModel):
    user_id: str
    gpt_id: str
    gpt_name: Optional[str] = "default_gpt"

class ChatPayload(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    # --- FIX: This is now deprecated but kept for frontend compatibility. The logic will ignore it. ---
    user_document_keys: Optional[List[str]] = Field([], alias="user_documents")
    use_hybrid_search: Optional[bool] = False
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    web_search_enabled: Optional[bool] = False
    mcp_enabled: Optional[bool] = False
    mcp_schema: Optional[str] = None
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

class ChatStreamRequest(BaseAgentRequest, ChatPayload):
    memory: Optional[List[Dict[str, str]]] = []

class ChatRequest(BaseAgentRequest, ChatPayload):
    pass

class GptContextSetupRequest(BaseAgentRequest):
    kb_document_urls: Optional[List[str]] = []
    default_model: Optional[str] = None
    default_system_prompt: Optional[str] = None
    default_use_hybrid_search: Optional[bool] = False
    mcp_enabled_config: Optional[bool] = Field(None, alias="mcpEnabled")
    mcp_schema_config: Optional[str] = Field(None, alias="mcpSchema")
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

class FileUploadInfoResponse(BaseModel):
    filename: str
    stored_url_or_key: str
    status: str
    error_message: Optional[str] = None

class GptOpenedRequest(BaseModel):
    user_id: str
    gpt_id: str
    gpt_name: str
    file_urls: List[str] = []
    use_hybrid_search: bool = False
    web_search_enabled: bool = True
    config_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)


# --- Helper Functions (Merged and Refactored) ---

def get_session_id(user_id: str, gpt_id: str) -> str:
    """Generate session ID from user id and GPT ID (robust version)."""
    id_part = user_id.replace('@', '_').replace('.', '_')
    return f"user_{id_part}_gpt_{gpt_id}"

def sanitize_for_collection_name(name: str) -> str:
    """Sanitizes a string to be a valid Qdrant collection name."""
    # Replace any character that is not a letter, number, or underscore with an underscore.
    s = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it doesn't start or end with an underscore.
    s = s.strip('_')
    # Prepend a prefix to ensure uniqueness and validity.
    if not s:
        return "gpt_collection_default"
    return f"gpt_collection_{s}"

async def get_or_create_agent(
    user_id: str,
    gpt_id: str,
    gpt_name: Optional[str] = "default_gpt",
    api_keys: Optional[Dict[str, str]] = None,
    force_recreate: bool = False,
    **config_overrides
) -> ChatbotAgent:
    """
    Get or create a ChatbotAgent, ensuring it uses the single, shared storage client.
    If config changes are detected, it replaces the existing agent with a new one.
    """
    global r2_storage, global_llm_manager, global_qdrant_client
    # Generate a unique and valid collection name from the gpt_id.
    qdrant_collection_name = sanitize_for_collection_name(gpt_id)
    agent_key = f"{user_id}_{gpt_id}"
    
    # Check if an existing agent has the wrong collection name, and if so, force a recreate.
    existing_agent = active_agents.get(agent_key)
    if existing_agent and existing_agent.rag_pipeline.config.qdrant_collection_name != qdrant_collection_name:
        logger.info(f"Collection name mismatch for agent {agent_key}. Forcing recreation.")
        force_recreate = True
        
    async with agents_lock:
        if agent_key in active_agents and not force_recreate:
            return active_agents[agent_key]
        
        if agent_key in active_agents and force_recreate:
            logger.info(f"🔄 Configuration changed for agent {agent_key}. Recreating...")
            old_agent = active_agents.pop(agent_key)
            asyncio.create_task(old_agent.cleanup())
    
    try:
        logger.info(f"🤖 Instantiating FAST ChatbotAgent for {agent_key} with collection '{qdrant_collection_name}'")
        config = ChatbotConfig.from_env()

        if api_keys:
            config.openai_api_key = api_keys.get('openai', config.openai_api_key)
            config.anthropic_api_key = api_keys.get('claude', config.anthropic_api_key)
            config.google_api_key = api_keys.get('gemini', config.google_api_key)
            config.groq_api_key = api_keys.get('groq', config.groq_api_key)
            config.openrouter_api_key = api_keys.get('openrouter', config.openrouter_api_key)
            config.tavily_api_key = api_keys.get('tavily', config.tavily_api_key)
        
        config_overrides['qdrant_collection_name'] = qdrant_collection_name
        for key, value in config_overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required but was not provided.")

        # --- START: Pass global clients to the agent constructor ---
        agent = ChatbotAgent(
            config=config,
            storage_client=r2_storage,
            llm_manager=global_llm_manager,
            qdrant_client=global_qdrant_client
        )
        await agent.initialize()
        
        async with agents_lock:
            active_agents[agent_key] = agent
        
        logger.info(f"✅ ChatbotAgent created successfully for {agent_key}")
        return agent

    except Exception as e:
        logger.error(f"❌ Failed to create ChatbotAgent for {agent_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create chatbot agent: {str(e)}")

async def process_uploaded_file_to_r2(file: UploadFile, is_user_doc: bool) -> FileUploadInfoResponse:
    """Process uploaded file and store it in R2 using the shared client."""
    global r2_storage
    if not r2_storage or r2_storage.use_local_fallback:
        error_msg = "R2 storage is not available or in fallback mode."
        logger.error(f"❌ Failed to upload '{file.filename}': {error_msg}")
        return FileUploadInfoResponse(filename=file.filename, stored_url_or_key="", status="failure", error_message=error_msg)

    try:
        file_content = await file.read()
        
        success, r2_path_or_error = await asyncio.to_thread(
            r2_storage.upload_file,
            file_data=file_content,
            filename=file.filename,
            is_user_doc=is_user_doc
        )

        if success:
            logger.info(f"✅ File '{file.filename}' uploaded to R2: {r2_path_or_error}")
            return FileUploadInfoResponse(filename=file.filename, stored_url_or_key=r2_path_or_error, status="success")
        else:
            logger.error(f"❌ Failed to upload '{file.filename}' to R2: {r2_path_or_error}")
            return FileUploadInfoResponse(filename=file.filename, stored_url_or_key="", status="failure", error_message=r2_path_or_error)
            
    except Exception as e:
        logger.error(f"❌ Exception processing file '{file.filename}': {e}", exc_info=True)
        return FileUploadInfoResponse(filename=file.filename, stored_url_or_key="", status="failure", error_message=str(e))

# --- API Endpoints (Merged and Harmonized) ---

@app.get("/", include_in_schema=False)
async def root_redirect():
    return JSONResponse(content={"message": "Enhanced Chatbot API is running. Visit /docs for details."})

@app.get("/health", summary="Health check endpoint", tags=["Monitoring"])
async def health_check():
    global r2_storage
    storage_healthy = r2_storage and not r2_storage.use_local_fallback
    return {"status": "healthy", "timestamp": time.time(), "agent_available": AGENT_AVAILABLE, "storage_healthy": storage_healthy}


@app.post("/setup-gpt-context", summary="Initialize/update a GPT's context (Compatible)", tags=["Agent Setup"])
async def setup_gpt_context_endpoint(request: GptContextSetupRequest, background_tasks: BackgroundTasks):
    logger.info(f"🔧 Received setup request for GPT ID: {request.gpt_id}")
    try:
        agent = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys,
            force_recreate=True,
            default_model=request.default_model,
            default_system_prompt=request.default_system_prompt,
            use_hybrid_search=request.default_use_hybrid_search,
        )
        
        async def _process_kb_urls_task(urls: List[str], agent_instance: ChatbotAgent):
            if urls:
                logger.info(f"📚 BG Task: Updating KB for agent with {len(urls)} URLs.")
                # Using the corrected method name from agent.py
                await agent_instance.add_documents_to_knowledge_base(urls)
                logger.info(f"✅ BG Task: KB update finished for agent.")
        
        if request.kb_document_urls:
            background_tasks.add_task(_process_kb_urls_task, request.kb_document_urls, agent)
            message = "Agent context update initiated in background."
        else:
            message = "Agent context initialized/updated with new defaults."

        collection_name = agent.rag_pipeline.config.qdrant_collection_name if agent.rag_pipeline else None
        return JSONResponse(content={"success": True, "message": message, "collection_name": collection_name})

    except Exception as e:
        logger.error(f"❌ Error in setup_gpt_context_endpoint for GPT {request.gpt_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to setup GPT context: {str(e)}")

@app.post("/upload-documents", summary="Upload documents (KB or User-specific)", tags=["Documents"])
async def upload_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    gpt_id: str = Form(...),
    is_user_document: str = Form("false"),
):
    is_user_doc_bool = is_user_document.lower() == "true"
    doc_type = "user-specific" if is_user_doc_bool else "knowledge base"
    logger.info(f"Uploading {len(files)} {doc_type} documents for GPT {gpt_id}")
    
    upload_tasks = [process_uploaded_file_to_r2(file, is_user_doc_bool) for file in files]
    results = await asyncio.gather(*upload_tasks)
    successful_uploads = [res.stored_url_or_key for res in results if res.status == "success"]

    if not successful_uploads:
        return JSONResponse(status_code=400, content={"message": "No files were successfully uploaded.", "upload_results": [r.model_dump() for r in results]})

    agent = await get_or_create_agent(user_id=user_id, gpt_id=gpt_id)
    
    # --- START FIX: Index documents statefully on the backend ---
    async def _index_documents_task(agent_instance: ChatbotAgent, urls: List[str], is_user_specific: bool, session_id: str):
        if is_user_specific:
            logger.info(f"👤 BG Task: Indexing {len(urls)} user documents for session {session_id}...")
            await agent_instance.add_user_documents_for_session(session_id, urls)
            logger.info(f"✅ BG Task: Indexing complete for user documents.")
        else:
            logger.info(f"📚 BG Task: Indexing {len(urls)} KB documents...")
            await agent_instance.add_documents_to_knowledge_base(urls)
            logger.info(f"✅ BG Task: Indexing complete for KB documents.")

    session_id = get_session_id(user_id, gpt_id)
    background_tasks.add_task(_index_documents_task, agent, successful_uploads, is_user_doc_bool, session_id)
    
    message = f"{len(successful_uploads)} {doc_type} files accepted and are being indexed in the background."
    if is_user_doc_bool:
        message += " They will be automatically available for your chat session."
    # --- END FIX ---

    return JSONResponse(status_code=202, content={
        "message": message,
        "upload_results": [r.model_dump() for r in results]
    })


@app.post("/chat-stream", summary="Streaming chat with agent", tags=["Chat"])
async def chat_stream(request: ChatStreamRequest):
    try:
        session_id = get_session_id(request.user_id, request.gpt_id)
        logger.info(f"💬 Chat stream request from {request.user_id} for GPT {request.gpt_id}")
        
        # --- START FIX: Log based on stateful logic ---
        logger.info("📄 Agent will automatically use any indexed user-specific documents for this session.")
        if request.user_document_keys:
             logger.warning("⚠️ 'user_documents' field is deprecated. Documents should be uploaded via /upload-documents to be indexed for the session.")
        # --- END FIX ---

        agent = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys,
            force_recreate=bool(request.api_keys)
        )
        
        async def generate():
            try:
                # --- START FIX: Remove user_documents from the call ---
                # The agent now retrieves them automatically from the indexed session collection.
                async for chunk in agent.query_stream(
                    session_id=session_id,
                    query=request.message,
                    enable_web_search=request.web_search_enabled,
                    model_override=request.model,
                    system_prompt_override=request.system_prompt,
                    mcp_enabled=request.mcp_enabled,
                    mcp_schema=request.mcp_schema,
                    api_keys=request.api_keys
                ):
                # --- END FIX ---
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                logger.error(f"❌ Error during stream generation for session {session_id}: {e}", exc_info=True)
                error_chunk = {"type": "error", "data": f"Stream generation error: {str(e)}"}
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
                    
    except Exception as e:
        logger.error(f"❌ Error in chat stream endpoint: {e}", exc_info=True)
        error_message = f"Failed to start chat stream: {str(e)}"
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'data': error_message})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=500)

@app.post("/chat", summary="Non-streaming chat with agent", tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = get_session_id(request.user_id, request.gpt_id)
        logger.info(f"💬 Non-streaming chat request from {request.user_id} for GPT {request.gpt_id}")
        
        agent = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys,
            force_recreate=bool(request.api_keys)
        )
        
        # --- START FIX: Remove user_documents from the call ---
        response = await agent.query(
            session_id=session_id,
            query=request.message,
            enable_web_search=request.web_search_enabled,
            model_override=request.model,
            system_prompt_override=request.system_prompt,
            mcp_enabled=request.mcp_enabled,
            mcp_schema=request.mcp_schema,
            api_keys=request.api_keys
        )
        # --- END FIX ---
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {str(e)}"})

@app.post("/gpt-opened", summary="Notify backend when a GPT is opened, ensure context is set up.")
async def gpt_opened_endpoint(request: GptOpenedRequest, background_tasks: BackgroundTasks):
    session_id = get_session_id(request.user_id, request.gpt_id)
    logger.info(f"GPT opened: ID={request.gpt_id}, Name='{request.gpt_name}', User={request.user_id}")

    try:
        force_recreate = bool(request.api_keys)
        agent_instance = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.config_schema.get("model") if request.config_schema else None,
            default_system_prompt=request.config_schema.get("instructions") if request.config_schema else None,
            use_hybrid_search=request.use_hybrid_search,
            api_keys=request.api_keys,
            force_recreate=force_recreate
        )
        
        if request.file_urls:
            async def process_documents(agent: ChatbotAgent, urls: List[str]):
                logger.info(f"Processing {len(urls)} document URLs for KB in background")
                try:
                    await agent.add_documents_to_knowledge_base(urls)
                    logger.info("KB document processing completed successfully")
                except Exception as e:
                    logger.error(f"Error processing KB documents in background: {e}", exc_info=True)
            background_tasks.add_task(process_documents, agent_instance, request.file_urls)
        
        collection_name = agent_instance.rag_pipeline.config.qdrant_collection_name if agent_instance.rag_pipeline else None
        
        mcp_enabled = request.config_schema.get("mcpEnabled", False) if request.config_schema else False
        mcp_schema = request.config_schema.get("mcpSchema") if request.config_schema else None

        return JSONResponse(content={
            "success": True,
            "message": f"GPT '{request.gpt_name}' context initialized/updated for user {request.user_id}.",
            "collection_name": collection_name,
            "session_id": session_id,
            "mcp_config_loaded": {
                "enabled": mcp_enabled,
                "schema_present": bool(mcp_schema)
            }
        })
    except Exception as e:
        logger.error(f"Error in gpt_opened_endpoint for GPT {request.gpt_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to handle GPT opened event: {str(e)}")

@app.post("/upload-chat-files", summary="Upload files for chat (Compatible)", tags=["Documents"])
async def upload_chat_files_endpoint(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    gpt_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    logger.info(f"Uploading {len(files)} chat files for user {user_id}, GPT {gpt_id}")
    
    upload_tasks = [process_uploaded_file_to_r2(file, is_user_doc=True) for file in files]
    results = await asyncio.gather(*upload_tasks)
    
    successful_urls = [res.stored_url_or_key for res in results if res.status == "success"]
    
    if not successful_urls:
         return JSONResponse(status_code=400, content={
            "success": False,
            "message": "No files were successfully uploaded.",
            "file_urls": [],
            "processing_results": [r.model_dump() for r in results]
        })

    # --- START FIX: Index user documents statefully ---
    agent = await get_or_create_agent(user_id=user_id, gpt_id=gpt_id)
    session_id = get_session_id(user_id, gpt_id)

    async def _index_user_docs_task(agent_instance: ChatbotAgent, s_id: str, urls: List[str]):
        logger.info(f"👤 BG Task: Indexing {len(urls)} user chat files for session {s_id}...")
        await agent_instance.add_user_documents_for_session(s_id, urls)
        logger.info(f"✅ BG Task: Indexing complete for user chat files.")
    
    background_tasks.add_task(_index_user_docs_task, agent, session_id, successful_urls)
    # --- END FIX ---

    return JSONResponse(status_code=202, content={
        "success": True,
        "message": f"Processed {len(successful_urls)} files. They are being indexed and will be available for your chat session automatically.",
        "file_urls": successful_urls,
        "processing_results": [r.model_dump() for r in results]
    })


@app.post("/index-knowledge", summary="Index knowledge from URLs and schema (Compatible)", tags=["Documents"])
async def index_knowledge_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        gpt_id = data.get("gpt_id")
        user_id = data.get("user_id", "default_user")
        file_urls = data.get("file_urls", [])
        
        if not gpt_id or not file_urls:
            raise HTTPException(status_code=400, detail="gpt_id and file_urls are required.")
            
        logger.info(f"📚 Index knowledge request for GPT {gpt_id} with {len(file_urls)} URLs")
        
        agent = await get_or_create_agent(user_id=user_id, gpt_id=gpt_id)
        
        async def _process_kb_urls_task(urls: List[str], agent_instance: ChatbotAgent):
            logger.info(f"📚 BG Task: Processing {len(urls)} KB URLs for agent...")
            await agent_instance.add_documents_to_knowledge_base(urls)
            logger.info(f"✅ BG Task: Knowledge indexing complete.")

        background_tasks.add_task(_process_kb_urls_task, file_urls, agent)
        
        return {"success": True, "message": f"Indexing started for {len(file_urls)} files."}
    
    except Exception as e:
        logger.error(f"❌ Error in index-knowledge endpoint: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/dev/reset-gpt-context", summary="DEV ONLY: Clear agent context (Compatible)", tags=["Development"])
async def dev_reset_gpt_context_endpoint(gpt_id: str = Form(...), user_id: str = Form(...)):
    if os.getenv("ENVIRONMENT_TYPE", "production").lower() != "development":
        raise HTTPException(status_code=403, detail="Endpoint only available in development.")

    agent_key = f"{user_id}_{gpt_id}"
    async with agents_lock:
        if agent_key in active_agents:
            logger.info(f"DEV: Clearing context for agent '{agent_key}'")
            agent = active_agents.pop(agent_key)
            await agent.cleanup()
            return {"status": "success", "message": f"Agent context for '{agent_key}' cleared from memory."}
        else:
            return JSONResponse(status_code=404, content={"status": "not_found", "message": f"No active agent context for '{agent_key}'."})

@app.post("/maintenance/cleanup-r2", summary="Manually trigger cleanup of expired R2 files", tags=["Maintenance"])
async def manual_cleanup_r2():
    try:
        await cleanup_r2_expired_files()
        return {"status": "success", "message": "R2 cleanup task triggered successfully."}
    except Exception as e:
        logger.error(f"❌ Error during manual R2 cleanup: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    if not AGENT_AVAILABLE:
        logger.critical("❌ ChatbotAgent is not available. The application cannot start.")
        sys.exit(1)
        
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=os.getenv("ENVIRONMENT_TYPE", "production").lower() == "development"
    )