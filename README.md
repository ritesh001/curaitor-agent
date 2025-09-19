# Curaitor Agent  
[![Documentation Status]] (https://curaitor-agent-docs.readthedocs.io/latest/)
**AI agent for scientific data extraction**  
Part of Schmidt OxRSE Workshop (Sep 11–20, 2025)  

---

## Overview  
Curaitor Agent is an **AI-powered tool** designed to extract, organize, and process **scientific data**.  
It provides:  
- A **web interface** for running the agent.  
- **Model Context Protocol (MCP) inspector** integration to test tools and server connections.  

## Documentation
https://curaitor-agent-docs.readthedocs.io/latest/

---

## Quick Start  

```bash
#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### Clone repo
```bash
git clone git@github.com:ritesh001/curaitor-agent.git
cd curaitor-agent
```
#### Initialize project
```bash
uv init
```

#### Edit config file
choose the model you want to use under llm:
  - provider: openai
  - model: "gpt-5-mini"

#### add .env file
Create .env file in the agent folder with your 
OPENAI_API_KEY=
OPENROUTER_API_KEY=

#### If you want to use email function 
- send your gmail email address to nsophonrat2@gmail.com to be added to the user pool

#### Run web interface
```bash
uv run adk web
```

### Functions you can use
#### curaitor_agent
- create database
- query database

#### curaitor_agent_v2
- search and summarize paper from arxiv
- schedule time of day for daily search
- send email summary to yourself
   - send email to nsophonrat2@gmail.com to be added to the user pool

---

## For Developer
### Dependency Management  

- Sync when `requirements.txt` is updated:  
  ```bash
  uv sync
  ```

- Add a new package:  
  ```bash
  uv add package-name
  ```
  *(Don’t forget to update `requirements.txt`!)*

---

### MCP Inspector Tool  

The **MCP Inspector** helps verify your MCP server connection and test available tools.  

### Requirements  
- [nvm](https://github.com/nvm-sh/nvm) (Node Version Manager)  
- **Node.js ≥ 18** (v22 recommended)  

### Setup  

1. Install **nvm**:  
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
   \. "$HOME/.nvm/nvm.sh"
   ```

2. Install **Node.js v22**:  
   ```bash
   nvm install 22
   ```

3. Verify versions:  
   ```bash
   node -v   # v22.19.0
   npm -v    # 10.9.3
   ```

4. Run the MCP Inspector:  
   ```bash
   npx @modelcontextprotocol/inspector uv run tools/mcp_server.py
   ```

5. In the MCP Inspector UI, click **Connect** → test tools.

---

## Notes  
- Ensure you’re using **Node.js v22.x** when running the inspector.  
- Always keep your environment in sync with `requirements.txt` for reproducibility.  

---

## License  
This project is licensed under the **MIT License**.  
