# curaitor-agent
AI agent for scientific data extraction
Part of Schmidt OxRSE workshop Sep 11-20, 2025

## Team members



# Installation
- Install uv 
`curl -LsSf https://astral.sh/uv/install.sh | sh`

- Clone the repo, then `cd` into the repo
- Initialize the project
`uv add -r requirements.txt` or `uv sync`
- Run agent web interface
`uv run adk web`


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

