# agent file based on Google ADK with MCP tool integration
# Nanta, Shichuan
# Sep 2025

import os
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
# from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
# from custom_adk_patches import CustomMCPToolset as MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
import yaml
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tool')))
# from agent_search_chunk import _normalize_model_for_provider

# models
config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

provider = config["llm"][0]["provider"]
# use_model = config["llm"][1]["model"]

# print(f"openrouter/{use_model}")

# if provider == "openai":
#     model = LiteLlm(
#         model=use_model,
#         api_key=os.getenv("OPENAI_API_KEY"),
#         api_base="https://api.openai.com/v1"
#     )
# if provider == "openrouter":
#     model = LiteLlm(
#         model=f"openrouter/{use_model}",
#         api_key=os.getenv("OPENROUTER_API_KEY"),
#         api_base="https://openrouter.ai/api/v1"
#     )
# else:
#     print(f"Provider {provider} is not supported.")

raw_model = config['llm'][1]['model']

if provider == 'openrouter':
    api_key = os.getenv("OPENROUTER_API_KEY")
elif provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
elif provider == "google":
    api_key = os.getenv("GOOGLE_API_KEY")
else:
    raise ValueError(f"Unsupported LLM provider: {provider}")
if not api_key:
    raise ValueError(f"Missing API key for provider '{provider}'. Set the appropriate env var.")

def _normalize_model_for_provider(p: str, m: str | None) -> str | None: ## can remove if all python files are in one folder
    if not m:
        return None
    m = m.strip()
    # For direct providers, drop OpenRouter-style prefixes/suffixes
    if p in ("openai", "google"):
        if "/" in m:
            m = m.split("/")[-1]  # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
        if ":" in m:
            m = m.split(":")[0]   # e.g., "...:free" -> base model
    return m

LLM_MODEL = _normalize_model_for_provider(provider, raw_model)

# Reasonable fallbacks if model omitted
if not LLM_MODEL:
    LLM_MODEL = {
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-flash",
        "openrouter": "google/gemini-2.0-flash-exp:free",
    }.get(provider, None)
print(f"[INFO] Provider: {provider} | Model: {LLM_MODEL}")

def _to_adk_model(p: str, llm: str, raw: str) -> str:
    if p == "google":
        return f"gemini/{llm}"
    # For OpenAI/OpenRouter we’ll pass a LiteLlm instance instead of a string.
    return f"{p}/{llm}"

# Keep this for Gemini only
ADK_MODEL = _to_adk_model(provider, LLM_MODEL, raw_model)

# Build model for Agent:
if provider == "google":
    model_for_agent = ADK_MODEL  # ADK natively supports gemini/*
elif provider == "openai":
    model_for_agent = LiteLlm(
        model=f"openai/{LLM_MODEL}",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1",
    )
elif provider == "openrouter":
    model_for_agent = LiteLlm(
        model=f"openrouter/{raw_model}",  # keep exact OpenRouter tag
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
    )
else:
    raise ValueError(f"Unsupported provider for ADK: {provider}")

# MCP toolset
mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        # test/ mock-up functions
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "curaitor_agent/mcp_server.py"],
            env=os.environ.copy(),
        ),
        # server for sending messages
        # server schedule tracker
    ),
    # optional: only expose specific tools
    # tool_filter=["my_tool_a", "my_tool_b"],
    # increase timeouts to tolerate slower startup/responses
    # startup_timeout_seconds=30,   # handshake/init
    # request_timeout_seconds=60,   # individual request (initialize, list_tools, etc.)
)

data_initializer_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        # test/ mock-up functions
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "curaitor_agent/curaitor_mcp_server.py"],
            env=os.environ.copy(),
        ),
        # server for sending messages
        # server schedule tracker
    ),
    # optional: only expose specific tools
    # tool_filter=["my_tool_a", "my_tool_b"],
        # increase timeouts to tolerate slower startup/responses
    # startup_timeout_seconds=30,   # handshake/init
    # request_timeout_seconds=60,   # individual request (initialize, list_tools, etc.)
)

root_agent = Agent(
    name="literature_agent",
    model=model_for_agent,
    description=(
        "Agent to track and summarize literature."
    ),
    instruction=(
        "You are an intelligent assistant capable of using external tools via MCP."
    ),
    tools=[data_initializer_mcp]
)