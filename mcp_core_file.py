import os
import re
import json
import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Tuple
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from llm_model import LLMManager

from dotenv import load_dotenv
load_dotenv()

# Pydantic is optional but recommended for tool validation
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.warning("Pydantic not available. Some validation features will be disabled.")
    # Define a fallback class if Pydantic isn't available
    class BaseModel:
        pass

# LangChain is optional, used for Document objects if available
try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Define a fallback Document class if LangChain isn't available
    class Document:
        def __init__(self, page_content: str = "", metadata: Optional[Dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Centralized LLM Manager (optional, for LLM-based analysis)
try:
    from llm_model import LLMManager, LLMConfig
    LLM_MODEL_AVAILABLE = True
except ImportError:
    LLM_MODEL_AVAILABLE = False
    logging.error("llm_model.py not found. LLM-based analysis features will be disabled.")
    # Define fallback classes if llm_model.py is not available
    class LLMManager:
        def __init__(self, config=None):
            logging.warning("LLMManager functionality is disabled.")
        async def generate_response(self, **kwargs):
            return ""
    class LLMConfig:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MCPConfiguration:
    """Configuration class for the MCP Core Handler."""
    
    # API Keys for LLM integration (used for analysis functions)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    
    # LLM settings for intelligent query analysis and tool argument construction
    use_llm_for_analysis: bool = True
    analysis_model: str = "gpt-4o-mini"
    analysis_temperature: float = 0.2
    
    # Timeout settings for subprocesses and network calls
    default_timeout: int = 120  # Increased default timeout for long-running tools
    json_read_timeout: float = 120.0 # Increased timeout for initial JSON-RPC handshake
    process_termination_timeout: float = 5.0 # Graceful termination period
    
    # Enhanced MCP server detection patterns, ordered by priority
    mcp_server_patterns: List[str] = field(default_factory=lambda: [
        r'@([a-zA-Z0-9\-_]+)(?:\s|$|:)',  # Catches @servername, @servername:, @servername<end_of_string>
        r'@\{([^}]+)\}',                    # Catches @{servername}
    ])
    
    # MODIFIED URL DETECTION PATTERNS ---
    # URL detection patterns for identifying URLs in queries
    url_patterns: List[str] = field(default_factory=lambda: [
        r'https?://[^\s/$.?#].[^\s]*',  # Full URLs
        r'www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?', # www. domains
        # Added a new, more robust pattern for standalone domains like 'google.com'
        r'\b[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|ai|dev|us|uk|ca|info|biz)\b(?:/[^\s]*)?'
    ])
    # MODIFIED URL DETECTION PATTERNS ---
    
    # Settings for keeping navigation tools (like browsers) alive between queries
    keep_navigation_tool_open: bool = True
    navigation_tool_names: List[str] = field(default_factory=lambda: [
        "navigate", "browser", "puppeteer", "playwright", "visit", "go", "open","fetch", "web", "http", "https", "url"
    ])
    
    def to_llm_config(self) -> Optional[LLMConfig]:
        """Converts MCPConfiguration to LLMConfig for the LLM Manager."""
        if not LLM_MODEL_AVAILABLE:
            return None
        return LLMConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            openrouter_api_key=self.openrouter_api_key,
            default_model=self.analysis_model,
            temperature=self.analysis_temperature
        )

    @classmethod
    def from_env(cls) -> 'MCPConfiguration':
        """Creates a configuration instance from environment variables."""
        return cls()


class MCPServerQueryTool(BaseModel):
    """A Pydantic model for validating the structure of an MCP server query tool call."""
    query_type: str = Field(default="mcp", description="Indicates this is an MCP query.")
    server_name: str = Field(description="Name of the MCP server to be used.")
    explanation: str = Field(description="An explanation of why this query should be routed to an MCP server.")


class MCPCore:
    """
    Handles the complete lifecycle of Model Context Protocol (MCP) requests. This class is responsible for
    detecting MCP-related queries, selecting the appropriate server, executing the server as a subprocess,
    managing the JSON-RPC communication, and handling the lifecycle of the subprocess.
    """
    
    def __init__(self, config: Optional[MCPConfiguration] = None, llm_manager: Optional[LLMManager] = None):
        """
        Initializes the MCP Core handler.
        
        Args:
            config: An MCPConfiguration object. If not provided, a default configuration is loaded.
            llm_manager: A pre-initialized LLMManager instance.
        """
        self.config = config or MCPConfiguration.from_env()
        
        # Tracks active MCP subprocesses for graceful cleanup
        self.active_mcp_processes: Dict[str, asyncio.subprocess.Process] = {}
        self.mcp_cleanup_lock = asyncio.Lock()
        
        # Initialize the centralized LLM Manager for query analysis if available
        self.llm_manager = llm_manager
        if not self.llm_manager:
            logger.warning("No global LLM Manager provided to MCPCore, creating a local instance.")
            if LLM_MODEL_AVAILABLE:
                llm_config = self.config.to_llm_config()
                if llm_config:
                    self.llm_manager = LLMManager(llm_config)
            else:
                logger.warning("⚠️ LLM-based analysis features are disabled (llm_model.py not found).")

        self.storage_client = None

    async def execute_mcp_request(
        self,
        query: str,
        mcp_schema: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        detected_server_name: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes a full MCP request cycle based on a user query and a schema of available servers.
        
        This is the primary entry point for handling an MCP query. It orchestrates server selection,
        environment preparation, and subprocess execution.
        
        Args:
            query: The user's query string.
            mcp_schema: A JSON string defining the available MCP servers.
            chat_history: A list of previous conversation turns for context.
            api_keys: A dictionary of API keys to be securely injected into the MCP server's environment.
            detected_server_name: The name of the server to use, if pre-determined.
            model_override: The specific model to use for any internal LLM calls.
            
        Yields:
            A dictionary representing a chunk of the response, with a 'type' ('content', 'error', 'done')
            and 'data'.
        """
        server_name = None # Start with no server name
        try:
            # Parse the provided MCP schema
            full_mcp_config = json.loads(mcp_schema)
            
            # Ensure the schema is in the expected format
            if "mcpServers" not in full_mcp_config or not isinstance(full_mcp_config["mcpServers"], dict):
                raise ValueError("Invalid MCP schema: Must contain an 'mcpServers' dictionary.")
            
            all_servers = full_mcp_config["mcpServers"]
            logger.info(f"🔍 Available MCP servers in schema: {list(all_servers.keys())}")

            # Determine which server to use
            if detected_server_name:
                logger.info(f"🎯 Agent pre-selected server: '{detected_server_name}'")
                # Fuzzy match the pre-selected name against available servers for robustness
                server_name = self._fuzzy_match_server_name(detected_server_name, all_servers)
            
            if not server_name:
                # If no server is pre-selected or the fuzzy match failed, detect from query
                server_name = await self._detect_mcp_server_from_query(query, all_servers, model_override=model_override)

            if not server_name or server_name not in all_servers:
                # If no specific server could be determined, fallback to the first one in the schema
                server_name = next(iter(all_servers.keys()), None)
                if server_name:
                    logger.info(f"🔄 No specific server detected, falling back to first available: '{server_name}'")
                else:
                    raise ValueError("No MCP servers available in the provided schema.")

            logger.info(f"🚀 Executing with MCP Server: '{server_name}' for query: '{query[:70]}...'")
            
            server_config = all_servers[server_name]
            if "command" not in server_config or not server_config["command"]:
                raise ValueError(f"Server configuration for '{server_name}' is missing a 'command'.")

            # Prepare environment variables for the subprocess
            effective_env = os.environ.copy()
            # 1. Add variables from the server's own config
            if "env" in server_config and isinstance(server_config["env"], dict):
                effective_env.update(server_config["env"])
            # 2. Add API keys provided in the request, but only if they are expected by the server config
            if api_keys:
                expected_keys = server_config.get("env", {}).keys()
                keys_to_add = {k: v for k, v in api_keys.items() if k in expected_keys}
                if keys_to_add:
                    logger.info(f"🔑 Injecting API keys into environment: {list(keys_to_add.keys())}")
                    effective_env.update(keys_to_add)

            # Execute the subprocess and stream its response
            async for response_chunk_str in self._execute_mcp_server_subprocess(
                server_name=server_name, 
                server_config=server_config, 
                query=query,
                chat_history=chat_history or [],
                environment=effective_env,
                model_override=model_override
            ):
                yield {"type": "content", "data": response_chunk_str}
            
            yield {"type": "done", "data": f"MCP execution for '{server_name}' completed."}

        except (json.JSONDecodeError, ValueError) as e_schema:
            error_msg = f"Invalid MCP Configuration: {str(e_schema)}"
            logger.error(error_msg)
            yield {"type": "error", "data": error_msg}
        except Exception as e:
            error_msg = f"Failed to process MCP request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield {"type": "error", "data": error_msg}

    def _fuzzy_match_server_name(self, detected_name: str, available_servers: Dict[str, Any]) -> Optional[str]:
        """
        Finds the best matching server name from a list of available servers using fuzzy matching.
        This allows for variations like '@perplexity' to match 'perplexity-ask-server'.
        """
        detected_lower = detected_name.lower().strip()
        
        # 1. Exact case-insensitive match
        for server_name in available_servers:
            if server_name.lower() == detected_lower:
                return server_name
        
        # 2. Check if detected name is a full part of a compound server name (e.g., 'perplexity' in 'perplexity-ask')
        for server_name in available_servers:
            server_parts = re.split(r'[-_]', server_name.lower())
            if detected_lower in server_parts:
                return server_name
        
        # 3. Check if detected name is a substring of any server name
        for server_name in available_servers:
            if detected_lower in server_name.lower():
                return server_name
        
        logger.warning(f"Could not find a fuzzy match for '{detected_name}' in {list(available_servers.keys())}")
        return None

    async def _detect_mcp_server_from_query(self, query: str, available_servers: Dict[str, Any], model_override: Optional[str] = None) -> Optional[str]:
        """Detects the intended MCP server from the query text using regex and optional LLM analysis."""
        # 1. Check for explicit @-mentions using regex patterns
        for pattern in self.config.mcp_server_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                detected_name = matches[0].strip()
                matched_server = self._fuzzy_match_server_name(detected_name, available_servers)
                if matched_server:
                    logger.info(f"✅ MCP server detected via @-pattern: '{detected_name}' -> '{matched_server}'")
                    return matched_server
        
        # 2. If LLM analysis is enabled, use it for intelligent selection
        if self.llm_manager and self.config.use_llm_for_analysis:
            return await self._intelligent_server_selection(query, list(available_servers.keys()), model_override=model_override)
        
        # 3. Fallback: check if a server name is mentioned plainly in the query
        query_lower = query.lower()
        for server_name in available_servers:
            if server_name.lower() in query_lower:
                 logger.info(f"✅ MCP server detected via name mention: '{server_name}'")
                 return server_name

        logger.info("ℹ️ No specific MCP server detected in query.")
        return None

    async def _execute_mcp_server_subprocess(
        self, 
        server_name: str, 
        server_config: Dict[str, Any], 
        query: str, 
        chat_history: List[Dict[str, str]],
        environment: Dict[str, str],
        model_override: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Handles the creation and management of the MCP server subprocess."""
        command = server_config.get("command")
        args = server_config.get("args", [])
        
        if not command:
            raise ValueError(f"No command specified for MCP server '{server_name}'")
        
        logger.info(f"Executing MCP server '{server_name}' with command: '{command}' and args: {args}")
        
        # Use a more robust communication protocol handler
        async for response_chunk in self._execute_and_communicate(command, args, environment, query, chat_history, model_override=model_override):
            yield response_chunk
                
    def _is_valid_json_line(self, line: str) -> bool:
        """Utility to check if a string is a valid JSON object."""
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                json.loads(line)
                return True
            except json.JSONDecodeError:
                return False
        return False

    async def _read_json_response(self, stdout_stream, timeout: float) -> Optional[Dict[str, Any]]:
        """Reads and parses a single JSON-RPC response from the process stdout with a timeout."""
        try:
            while True: # Loop to skip non-JSON informational lines
                line_bytes = await asyncio.wait_for(stdout_stream.readline(), timeout=timeout)
                if not line_bytes: return None
                
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if line and self._is_valid_json_line(line):
                    return json.loads(line)
                elif line:
                    logger.info(f"MCP server stdout (skipped): {line}")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for JSON response from MCP server after {timeout}s.")
            return None
        except Exception as e:
            logger.error(f"Error reading MCP server response stream: {e}")
            return None

    async def _execute_and_communicate(
        self, 
        command: str, 
        args: List[str], 
        env_vars: Dict[str, str], 
        query: str, 
        chat_history: List[Dict[str, str]],
        model_override: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        The core logic for running the server subprocess and handling the JSON-RPC communication protocol.
        This version uses `create_subprocess_shell` for robust, cross-platform command execution.
        """
        process = None
        session_id = f"mcp_{hash(query)}_{int(time.time())}"
        tool_name = "unknown" # Default tool name
        
        try:
            # Construct a single command string for the shell to execute.
            # This is more robust, especially on Windows for .cmd/.bat scripts like npx.
            command_string = f"{command} {' '.join(args)}"
            
            logger.info(f"🚀 Executing command via shell: {command_string}")

            # Create the subprocess using the system shell
            process = await asyncio.create_subprocess_shell(
                command_string,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars
            )
            self.active_mcp_processes[session_id] = process

            # Start JSON-RPC Communication ---
            
            # 1. Send `initialize` request
            init_request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            process.stdin.write((json.dumps(init_request) + "\n").encode())
            await process.stdin.drain()
            init_response = await self._read_json_response(process.stdout, self.config.json_read_timeout)
            
            #  MODIFIED TIMEOUT HANDLING ---
            if not init_response:
                # Attempt to read from stderr to provide more context for the failure
                try:
                    stderr_output_bytes = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                    stderr_output = stderr_output_bytes.decode(errors='ignore').strip()
                    error_detail = f"Stderr from server: {stderr_output}" if stderr_output else "No error output from server."
                except asyncio.TimeoutError:
                    error_detail = "Could not read error output from server."

                yield f"❌ MCP Initialization Failed: The server took too long to respond. {error_detail}\n\nPlease check the server's configuration or try a different query."
                # The `finally` block will now handle process termination.
                return 
            # MODIFIED TIMEOUT HANDLING ---

            logger.info(f"✅ MCP Server Initialized. Info: {init_response.get('result', {}).get('serverInfo', {})}")

            # 2. Send `initialized` notification
            initialized_notification = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
            process.stdin.write((json.dumps(initialized_notification) + "\n").encode())
            await process.stdin.drain()

            # 3. `tools/list` to see what the server offers
            list_tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            process.stdin.write((json.dumps(list_tools_request) + "\n").encode())
            await process.stdin.drain()
            tools_response = await self._read_json_response(process.stdout, self.config.json_read_timeout)
            if not tools_response or "result" not in tools_response or "tools" not in tools_response["result"]:
                yield "❌ Error: Failed to get a valid list of tools from the MCP server."
                return
            
            available_tools = tools_response["result"]["tools"]
            if not available_tools:
                yield "⚠️ MCP server reported no available tools."
                return
            logger.info(f"🔧 Available Tools: {[tool.get('name', 'unnamed') for tool in available_tools]}")

            # 4. Select the best tool and construct arguments
            selected_tool = self._select_best_tool_for_query(query, available_tools)
            tool_name = selected_tool.get("name", "unknown")
            logger.info(f"🎯 Selected Tool: '{tool_name}'")
            tool_arguments = await self._construct_tool_arguments(selected_tool, query, chat_history, model_override=model_override)

            # 5. `tools/call` to execute the main task
            tool_call_request = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": tool_name, "arguments": tool_arguments}}
            tool_call_json = json.dumps(tool_call_request) + "\n"
            logger.info(f"📞 Calling tool with: {tool_call_json.strip()}")
            process.stdin.write(tool_call_json.encode())
            await process.stdin.drain()

            # 6. Stream the response from the tool call
            while True:
                response = await self._read_json_response(process.stdout, self.config.default_timeout)
                if not response:
                    logger.info(f"MCP server stream ended or timed out for PID {process.pid}.")
                    break

                # Case 1: Handle streaming content notifications from the server
                if "method" in response and response["method"] == "text/content":
                    content = response.get("params", {}).get("content", "")
                    if content:
                        yield str(content)
                    continue # Continue listening for more chunks

                # Case 2: Handle the final result of the tool call
                if "result" in response and response.get("id") == 3: # Final response to tool call
                    result_data = response["result"]
                    content_to_yield = ""
                    # The final result might also contain content, so we process it.
                    if isinstance(result_data, dict) and "content" in result_data:
                        content = result_data["content"]
                        if isinstance(content, list):
                            # Handle structured content like in the original code
                            content_to_yield = "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
                        else:
                            content_to_yield = str(content)
                    elif result_data: # Handle cases where the result is just a string or other primitive
                        content_to_yield = str(result_data)

                    if content_to_yield:
                         yield content_to_yield

                    logger.info(f"Received final result for tool call (id=3) for PID {process.pid}.")
                    break # End of this specific tool call

                # Case 3: Handle errors
                elif "error" in response:
                    error_msg = response["error"].get("message", str(response["error"]))
                    logger.error(f"MCP Server Error for PID {process.pid}: {error_msg}")
                    yield f"❌ MCP Server Error: {error_msg}"
                    break # Stop on error

                # Case 4: Handle other responses (like from 'initialize') that we don't need to yield
                elif "result" in response:
                    logger.info(f"Received non-final result (id={response.get('id')}): {str(response['result'])[:100]}...")

                # Otherwise, it might be a notification we don't handle, so we just loop.
            
            # JSON-RPC Communication ---
            
            is_nav_tool = self._is_navigation_tool(tool_name)
            is_url_query = bool(self._extract_urls_from_query(query))
            
            if is_nav_tool and is_url_query and self.config.keep_navigation_tool_open:
                yield "\n\n---\n"
                yield "🌐 Browser session started. It will remain open for further commands.\n"
                yield "💡 To close it, send a new message like 'close browser'."
                return
        
        except Exception as e:
            logger.error(f"Critical error during MCP execution: {e}", exc_info=True)
            yield f"❌ An unexpected error occurred while running the MCP server: {str(e)}"
        
        finally:
            if process and session_id in self.active_mcp_processes:
                is_nav_tool = self._is_navigation_tool(tool_name)
                is_url_query = bool(self._extract_urls_from_query(query))
                
                should_keep_alive = is_nav_tool and is_url_query and self.config.keep_navigation_tool_open
                
                if not should_keep_alive:
                    logger.info(f"🧹 Cleaning up process for session {session_id} (PID: {process.pid})")
                    await self._terminate_process(process)
                    async with self.mcp_cleanup_lock:
                        if session_id in self.active_mcp_processes:
                            del self.active_mcp_processes[session_id]
                else:
                    logger.info(f"🌐 Keeping navigation tool alive for session {session_id} (PID: {process.pid})")

    def _is_navigation_tool(self, tool_name: str) -> bool:
        """Determines if a tool is for web navigation based on its name."""
        if not tool_name: return False
        return any(indicator in tool_name.lower() for indicator in self.config.navigation_tool_names)

    # REPLACED METHOD WITH MORE ROBUST LOGIC ---
    async def _construct_tool_arguments(self, tool_to_use: Dict[str, Any], query: str, messages: List[Dict[str, str]], model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Intelligently constructs the arguments dictionary for a given tool call with robust URL handling.
        """
        input_schema = tool_to_use.get("inputSchema", {}).get("properties", {})
        required_params = tool_to_use.get("inputSchema", {}).get("required", [])
        tool_args = {}

        # Step 1: Detect URLs using the improved regex.
        detected_urls = self._extract_urls_from_query(query)

        # Step 2: Handle the 'url' parameter with priority.
        if "url" in input_schema:
            if detected_urls:
                tool_args["url"] = detected_urls[0]
                logger.info(f"✅ Found URL via regex: '{detected_urls[0]}'")
            elif "url" in required_params:
                # If a URL is required but not found by regex, try to construct one.
                logger.info(f"⚠️ URL is required, but none found via regex. Attempting to construct from query: '{query}'")
                potential_domain = next((word.strip('.,!?;:\'\"') for word in query.split() if '.' in word and '@' not in word), None)
                if potential_domain:
                    constructed_url = f"https://{potential_domain}"
                    if self._is_valid_url_structure(constructed_url):
                        tool_args["url"] = constructed_url
                        logger.info(f"✅ Constructed a plausible URL: '{constructed_url}'")
                    else:
                        logger.warning(f"Constructed URL '{constructed_url}' is invalid. Falling back.")
                else:
                    logger.warning("Could not construct a plausible URL from the query.")

        # Step 3: Handle other special-cased parameters (like chat history).
        if "messages" in input_schema:
            formatted_messages = []
            for msg in messages:
                role = "user" if msg.get("role") == "user" else "assistant"
                if formatted_messages and formatted_messages[-1]["role"] == role: continue
                formatted_messages.append({"role": role, "content": msg.get("content", "")})
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                formatted_messages.append({"role": "user", "content": query})
            else:
                formatted_messages[-1] = {"role": "user", "content": query}
            tool_args["messages"] = formatted_messages

        # Step 4: Use the remaining query for common text-based parameters.
        cleaned_query = query
        if "url" in tool_args:
            # Remove the found/constructed URL from the query to avoid redundancy.
            # This handles both `https://google.com` and `google.com`
            url_to_remove = tool_args["url"].replace("https://", "").replace("http://", "")
            cleaned_query = query.replace(url_to_remove, "").strip()
        
        for param_name in ["query", "question", "text", "input", "prompt"]:
            if param_name in input_schema and param_name not in tool_args:
                # Use the cleaned query if available and not empty, otherwise the original.
                tool_args[param_name] = cleaned_query if cleaned_query else query
                break
        
        # Step 5: Final fallback for required arguments that are still missing.
        for param in required_params:
            if param not in tool_args:
                logger.warning(f"Required parameter '{param}' not filled. Using query as a fallback.")
                tool_args[param] = query

        # If, after all logic, no arguments are filled, use the original query for the first parameter.
        if not tool_args and input_schema:
            first_param = next(iter(input_schema.keys()), None)
            if first_param:
                logger.warning(f"No specific arguments constructed. Using query for the first parameter '{first_param}'.")
                tool_args[first_param] = query

        return tool_args

        
    def _select_best_tool_for_query(self, query: str, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selects the most appropriate tool from a list based on the query content."""
        if not available_tools: return {}
        if len(available_tools) == 1: return available_tools[0]

        query_lower = query.lower()
        scored_tools = []
        
        # Score tools based on name and description matching
        for tool in available_tools:
            score = 0
            tool_name = tool.get("name", "").lower()
            description = tool.get("description", "").lower()
            
            if self._is_navigation_tool(tool_name) and self._extract_urls_from_query(query):
                score += 20 # High score for navigation tools when a URL is present
            
            if tool_name in query_lower:
                score += 15
            
            common_words = set(query_lower.split()) & set(description.split())
            score += len(common_words) * 2
            
            scored_tools.append((tool, score))
        
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        best_tool = scored_tools[0][0]
        logger.info(f"Tool Selection: '{best_tool.get('name', 'unknown')}' chosen with score {scored_tools[0][1]}.")
        return best_tool

    def _is_valid_url_structure(self, url: str) -> bool:
        """Validate if the URL has a proper structure"""
        try:
            parsed = urlparse(url)
            # Must have a scheme (http, https) and a network location (domain)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            # Domain must have at least one dot
            if '.' not in parsed.netloc:
                return False
            return True
        except:
            return False

    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extracts and cleans all URLs from a query string."""
        urls = []
        # Find all potential URLs using the configured regex patterns
        for pattern in self.config.url_patterns:
            urls.extend(re.findall(pattern, query))
        
        cleaned_urls = []
        for url in urls:
            # Strip common punctuation that might stick to the URL from surrounding text
            url = url.strip('.,!?;()[]"\'')
            
            # Add a protocol if it's missing, which is common for typed domains like 'www.example.com'
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Final validation to ensure it's a well-formed URL
            if self._is_valid_url_structure(url):
                cleaned_urls.append(url)
            
        return list(set(cleaned_urls)) # Return only the unique URLs found

    async def cleanup_processes(self, session_id: Optional[str] = None):
        """Gracefully terminates active MCP subprocesses."""
        async with self.mcp_cleanup_lock:
            procs_to_clean = list(self.active_mcp_processes.items())
            if session_id:
                procs_to_clean = [(sid, p) for sid, p in procs_to_clean if sid.startswith(session_id)]
            
            if not procs_to_clean: return

            logger.info(f"Cleaning up {len(procs_to_clean)} MCP processes...")
            tasks = [self._terminate_process(p) for sid, p in procs_to_clean]
            await asyncio.gather(*tasks, return_exceptions=True)

            for sid, p in procs_to_clean:
                if sid in self.active_mcp_processes:
                    del self.active_mcp_processes[sid]
            logger.info("Cleanup complete.")

    async def _terminate_process(self, process: asyncio.subprocess.Process):
        """The actual logic for terminating a single process."""
        if process.returncode is not None: return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=self.config.process_termination_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Process {process.pid} did not terminate gracefully. Killing.")
            process.kill()
            await process.wait()
        except Exception as e:
            logger.error(f"Error during process termination: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensures all processes are cleaned up when the context manager exits."""
        await self.cleanup_processes()

    async def _intelligent_server_selection(self, query: str, available_servers: List[str], model_override: Optional[str] = None) -> Optional[str]:
        """Uses an LLM to select the most appropriate MCP server for a given query."""
        if not self.llm_manager or not available_servers:
            return None
        
        try:
            servers_list_str = ", ".join(f"'{s}'" for s in available_servers)
            prompt = f"Given the user query, which of the following MCP servers is the most appropriate to use? Servers: {servers_list_str}. Respond with only the server name, nothing else.\n\nQuery: \"{query}\""
            
            # Use the provided model, or the config's analysis model, or fallback to 'gpt-4o-mini'
            model_to_use = model_override or self.config.analysis_model or "gpt-4o-mini"
            response = await self.llm_manager.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model=model_to_use
            )
            
            selected_server = response.strip().strip("'\"")
            
            # Validate that the LLM's choice is a valid server
            if selected_server in available_servers:
                logger.info(f"🤖 LLM selected server: '{selected_server}'")
                return selected_server
            else:
                logger.warning(f"LLM selected an invalid server ('{selected_server}'). Will fall back.")
                return None
        except Exception as e:
            logger.error(f"Error during intelligent server selection: {e}")
            return None

    async def _llm_select_server(self, query: str, available_servers: List[str], model_override: Optional[str] = None) -> str:
        """Use centralized LLM Manager to select the most appropriate server."""
        try:
            if not self.llm_manager:
                return available_servers[0] if available_servers else ""
            
            servers_list = ", ".join(available_servers)
            
            messages = [
                {"role": "system", "content": f"You are an MCP server selector. Given a query and list of available servers, select the most appropriate one. Available servers: {servers_list}. Respond with only the server name."},
                {"role": "user", "content": f"Which server is best for this query: '{query}'"}
            ]
            
            # Use the override, then config, then fallback
            model_to_use = model_override or self.config.analysis_model or "gpt-4o-mini"
            response = await self.llm_manager.generate_response(
                messages=messages,
                model=model_to_use,
                temperature=self.config.analysis_temperature
            )
            
            selected = response.strip()
            
            # Validate selection
            if selected in available_servers:
                return selected
            else:
                # Fallback to first available server
                return available_servers[0] if available_servers else ""
                
        except Exception as e:
            logger.error(f"Error in LLM server selection: {e}")
            return available_servers[0] if available_servers else ""

    async def clear_mcp_cache(self, prefix: str = "mcp_cache") -> Dict[str, Any]:
        """Clear cached MCP results - DISABLED as caching is removed"""
        logger.info("🚫 MCP caching is disabled - no cache to clear")
        return {
            "success": True,
            "deleted_count": 0,
            "message": "MCP caching is disabled"
        }

    def get_storage_status(self) -> Dict[str, Any]:
        """Get storage integration status"""
        if not self.storage_client:
            return {
                "available": False,
                "reason": "Storage client not initialized"
            }
        
        return {
            "available": True,
            "r2_enabled": not self.storage_client.use_local_fallback if hasattr(self.storage_client, 'use_local_fallback') else False,
            "local_fallback": self.storage_client.use_local_fallback if hasattr(self.storage_client, 'use_local_fallback') else True,
            "cache_enabled": False,  # Always False since caching is disabled
            "cache_expiry_hours": 0
        }

    def _should_close_navigation_tool(self, query: str) -> bool:
        """
        Detect if the user wants to close the browser/navigation tool.
        """
        query_lower = query.lower().strip()
        
        # Direct close commands
        close_patterns = [
            "close browser", "close the browser", "close window", "close tab",
            "stop browser", "quit browser", "exit browser", "end browser",
            "close puppeteer", "stop puppeteer", "quit puppeteer",
            "done with browser", "finished browsing", "stop browsing"
        ]
        
        return any(pattern in query_lower for pattern in close_patterns)

    async def close_navigation_tools(self, session_id: str = None) -> Dict[str, Any]:
        """
        Close navigation tools (puppeteer/browser) when user explicitly requests it.
        """
        closed_count = 0
        
        async with self.mcp_cleanup_lock:
            processes_to_close = []
            
            if session_id:
                # Close specific session's navigation tools
                for process_key, process in self.active_mcp_processes.items():
                    if process_key.startswith(session_id):
                        processes_to_close.append((process_key, process))
            else:
                # Close all navigation tools
                processes_to_close = list(self.active_mcp_processes.items())
            
            for process_key, process in processes_to_close:
                if process and process.returncode is None:
                    try:
                        logger.info(f"🔒 Closing navigation tool: {process_key}")
                        process.terminate()
                        await asyncio.wait_for(process.wait(), timeout=0.1)
                        closed_count += 1
                    except asyncio.TimeoutError:
                        logger.warning(f"🔪 Force killing navigation tool: {process_key}")
                        process.kill()
                        await process.wait()
                        closed_count += 1
                    except Exception as e:
                        logger.error(f"Error closing navigation tool {process_key}: {e}")
                    
                    # Remove from tracking
                    if process_key in self.active_mcp_processes:
                        del self.active_mcp_processes[process_key]
        
        return {
            "success": True,
            "closed_count": closed_count,
            "message": f"Closed {closed_count} navigation tool(s)"
        }

    # Add immediate cleanup option for navigation tools
    async def force_close_navigation_tools(self, session_id: str = None) -> Dict[str, Any]:
        """Force close navigation tools immediately without graceful shutdown."""
        closed_count = 0
        
        async with self.mcp_cleanup_lock:
            processes_to_close = []
            
            if session_id:
                processes_to_close = [
                    (k, v) for k, v in self.active_mcp_processes.items() 
                    if k.startswith(session_id)
                ]
            else:
                processes_to_close = list(self.active_mcp_processes.items())
            
            # Parallel force termination
            tasks = []
            for process_key, process in processes_to_close:
                if process and process.returncode is None:
                    task = asyncio.create_task(self._force_kill_process(process))
                    tasks.append((process_key, task))
            
            # Wait for all kills in parallel with very short timeout
            if tasks:
                await asyncio.gather(
                    *[task for _, task in tasks], 
                    return_exceptions=True
                )
                
                for process_key, _ in tasks:
                    if process_key in self.active_mcp_processes:
                        del self.active_mcp_processes[process_key]
                        closed_count += 1
        
        return {
            "success": True,
            "closed_count": closed_count,
            "message": f"Force closed {closed_count} processes"
        }

    async def _force_kill_process(self, process):
        """Force kill a process immediately."""
        try:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=0.3)
        except:
            pass  # Best effort

    async def _start_navigation_auto_cleanup(self, session_id: str, delay_seconds: int = 60):
        """Auto-cleanup navigation tools after a delay."""
        await asyncio.sleep(delay_seconds)
        
        # Check if session still has navigation tools
        navigation_processes = [
            k for k in self.active_mcp_processes.keys() 
            if k.startswith(session_id) and "navigate" in k.lower()
        ]
        
        if navigation_processes:
            logger.info(f"🧹 Auto-cleaning navigation tools for session {session_id}")
            await self.force_close_navigation_tools(session_id)


if __name__ == "__main__":
    asyncio.run()