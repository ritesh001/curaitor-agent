# agent file based on Google ADK with MCP tool integration

import os
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
import yaml

# models
config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

provider = config["llm"][0]["provider"]
use_model = config["llm"][1]["model"]

print(f"openrouter/{use_model}")

if provider == "openai":
    model = LiteLlm(
        model=use_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1"
    )
if provider == "openrouter":
    model = LiteLlm(
        model=f"openrouter/{use_model}",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1"
    )
else:
    print(f"Provider {provider} is not supported.")


# MCP toolset
mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        # test/ mock-up functions
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "tools/mcp_server.py"],
            env=os.environ.copy(),
        )
    )
)

mcp_toolset2 = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "tools/server_test_scheduler.py"],
            env=os.environ.copy(),
        )
    )
)


root_agent = Agent(
    name="literature_agent",
    model=model,
    description=(
        "Agent to track and summarize literature."
    ),
    instruction=(
        "You are an intelligent assistant capable of using external tools via MCP."
    ),
    tools=[mcp_toolset, mcp_toolset2],
)