import os
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# use_model = "deepseek"
use_model = "gpt-4o"

if use_model == "deepseek":
    model = LiteLlm(model="openrouter/deepseek/deepseek-r1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    api_base="https://openrouter.ai/api/v1"
    )
if use_model == "gpt-4o":
    model = LiteLlm(model="openai/gpt-4o",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    api_base="https://api.openai.com/v1"
    )

mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "tools/mcp_server.py"],
            env=os.environ.copy(),
        ),
    ),
    # optional: only expose specific tools
    # tool_filter=["my_tool_a", "my_tool_b"],
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
    tools=[mcp_toolset],
)