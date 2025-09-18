# curaitor-agent  
**AI agent for scientific data extraction**  
Part of Schmidt OxRSE Workshop (Sep 11–20, 2025)  

---

## Overview  
Curaitor Agent is an **AI-powered tool** designed to extract, organize, and process **scientific data**.  
It provides:  
- A **web interface** for running the agent.  
- **Model Context Protocol (MCP) inspector** integration to test tools and server connections.  

---

## Quick Start  

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone <your-repo-url>
cd curaitor-agent

# Initialize project
uv init
uv add -r requirements.txt

# Run web interface
uv run adk web

---

## Dependency management

Sync when requirements.txt is updated:
uv sync

Add a new package:
uv add package-name

MCP Inspector Tool

The MCP Inspector helps verify your MCP server connection and test available tools.

Requirements

nvm
 (Node Version Manager)

Node.js ≥ 18 (v22 recommended)

Setup

Install nvm:











If you update the requirements.txt, then do `uv sync` Or if add a package directly to uv do `uv add package-name` then also update the requirements.txt. 

# MCP Inspector tool
For inspecting if you have a good connection with the MCP server and testing tools in the server files.

- Requirements: nnvm, node.js>=18. Ref (see: https://nodejs.org/en/download)
    - Download and install nvm:
    `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash`
    - in lieu of restarting the shell
    `\. "$HOME/.nvm/nvm.sh"`
    - Download and install Node.js:
    `nvm install 22`
    - Verify the Node.js version:
    `node -v` # Should print "v22.19.0"
    - Verify npm version:
    `npm -v` # Should print "10.9.3".
- Run MCP inspector `npx @modelcontextprotocol/inspector uv run tools/mcp_server.py`
- In the MCP inspector, click connect and test tools

