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

print(f"{use_model}, {provider}")

if provider == "openai":
    print("Using OpenAI model")
    model = LiteLlm(
        model=use_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1"
    )
elif provider == "openrouter":
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
            args=["run", "curaitor_agent_v2/mcp_server.py"],
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
            """
    You are an intelligent assistant that can call MCP tools and then present clear, actionable results.

    ## Gmail tool behavior
    If ask to summarize and email the daily arXiv summary, you must call the Gmail tool to send the email.
    The input format for the Gmail tool is as follows (you must include "html": true if the body contains HTML):
    {
        "to": "
        "subject": "Daily arXiv summary",
        "body": "<p>Here’s the summary...</p>",
        "html": true
    }

    """
    ),
    tools=[mcp_toolset],
)