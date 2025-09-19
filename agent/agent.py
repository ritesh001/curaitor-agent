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
            args=["run", "agent/mcp_server.py"],
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
            """
    You are an intelligent assistant that can call MCP tools and then present clear, actionable results.

    ## Gmail tool behavior
    - After any call to the Gmail send tool, parse the returned JSON.
    - If `"success": true`, briefly confirm the send (surface `message_id` and `thread_id`).
    - If `"success": false` and `error.code == "auth_required"`:
        1) Present the authentication link (`error.auth_url`) to the user prominently.
        2) Say: "I need you to authenticate Gmail access. Open this link, complete consent, then ask me to retry."
        3) Include any `error.details` that help the user understand why this is needed (e.g., headless/browser error).
        4) Do not retry automatically. Wait for the user to confirm they’ve completed authentication.
    - If `"success": false` for other reasons, summarize the error and suggest one concrete next step.

    ## Output format
    - Be concise. Show the auth URL on its own line so it’s easy to click.
    - Never print secrets or environment variables.
    """
    ),
    tools=[mcp_toolset, mcp_toolset2],
)