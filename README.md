# Curaitor Agent  

[![CI](https://github.com/ritesh001/curaitor-agent/actions/workflows/ci.yml/badge.svg?style=plastic)](https://github.com/ritesh001/curaitor-agent/actions/workflows/ci.yml)
[![Scheduled](https://github.com/ritesh001/curaitor-agent/actions/workflows/curaitor-scheduled.yml/badge.svg?style=plastic)](https://github.com/ritesh001/curaitor-agent/actions/workflows/curaitor-scheduled.yml)

[![Stars](https://img.shields.io/github/stars/ritesh001/curaitor-agent?style=plastic)](https://github.com/ritesh001/curaitor-agent/stargazers)
[![Issues](https://img.shields.io/github/issues/ritesh001/curaitor-agent?style=plastic)](https://github.com/ritesh001/curaitor-agent/issues)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue?style=plastic)](pyproject.toml)
[![License](https://img.shields.io/github/license/ritesh001/curaitor-agent?style=plastic)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/ritesh001/curaitor-agent?style=plastic)](https://github.com/ritesh001/curaitor-agent/commits/main)

## AI agent for scientific data extraction
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
uv sync
```

#### Edit config file
choose the model you want to use under llm:
  - provider: openai
  - model: "gpt-5-mini"

#### Provide gmail address
- send your gmail email address to nsophonrat2@gmail.com to be added to the user pool

#### add .env file
Create .env file in the agent folder with your 
```bash
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GMAIL_CREDENTIALS_PATH=
GMAIL_TOKEN_PATH=secrets/token.json
```

#### Run gmail authentication
This will work for 1 hour.
```bash
uv run python curaitor_agent_v2/gmail_create_token.py
```

#### Run web interface
```bash
uv run adk web
```

#### Run LangGraph pipeline (no Google SDK)
This runs the literature RAG workflow orchestrated by LangGraph using your config and API keys.
```bash
uv run python -m curaitor_agent.langraph_pipeline --query "your research question"
```

## Scheduling (LangGraph)

The pipeline runs on **GitHub Actions** using the scheduled workflow in
`.github/workflows/curaitor-scheduled.yml`. You can:
- Use the built-in schedule (cron) for automatic runs.
- Trigger manually via **Actions → Curaitor LangGraph Weekly → Run workflow**.

Outputs are uploaded to S3 under:
`s3://curaitor-agent-dec2025/curaitor/<run_id>/`

The CLI wrapper `scripts/run_daily.py` runs the pipeline and upserts results into `data/curaitor.sqlite` by default.

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

### Notes  
- Ensure you’re using **Node.js v22.x** when running the inspector.  
- Always keep your environment in sync with `requirements.txt` for reproducibility.  

---

## License  
This project is licensed under the **MIT License**.  
