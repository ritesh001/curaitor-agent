
# This is a sample MCP server that provides tools to search for papers on arXiv
# and extract information about specific papers. It uses the `arxiv` library to
# Nanta, Shichuan
# Sep 2025
import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
# from arxiv_search_chunk import download_arxiv_pdfs
# from arxiv_search_chunk import get_keywords_from_llm
import yaml
import subprocess
import numpy as np
from pathlib import Path
import importlib.util
import requests
import time as time_module
# import argparse

config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
save_dir = config['source'][0]['pdf_path']
# PAPER_DIR = "data/agent_papers"
PAPER_DIR = save_dir

mcp = FastMCP("curaitor_mcp_server")

# def _run_arxiv_pipeline(query: str, max_days: int) -> List[str]:
#     """
#     Runs the arxiv_search_chunk pipeline with env overrides and returns arXiv IDs.
#     """
#     project_root = Path(__file__).resolve().parents[1]
#     env = os.environ.copy()
#     env["ARXIV_QUERY"] = query
#     env["ARXIV_MAX_DAYS"] = str(max_days)

#     # Execute the pipeline script
#     subprocess.run(
#         ["uv", "run", "tools/arxiv_search_chunk.py"],
#         check=True,
#         cwd=project_root,
#         env=env,
#     )

    # # Read the IDs from the NPZ saved by the script
    # npz_path = project_root / "arxiv_out_hits.npz"
    # if not npz_path.exists():
    #     return []
    # data = np.load(npz_path, allow_pickle=True)
    # ids = [str(aid) for aid in data["arxiv_ids"]]
    # return ids


# def input_query(query: str, max_days: int = 30, save_dir: str = None) -> List[str]:
#     """
#     Input a query to search for papers on arXiv based on a topic and store their information.

#     Args:
#         query: The topic to search for
#         max_days: Maximum age of papers in days (default: 7)
#         max_results: Maximum number of results to retrieve (default: 5)
#     Returns:
#         List of paper IDs found in the search
#     """

#     paper_ids = download_arxiv_pdfs(query, output_dir=save_dir)
#     return paper_ids
# @mcp.tool()
# # def input_query(query: str, max_days: int = 30, save_dir: str = None) -> List[str]:
# def input_query(query: str) -> None:
#     """
#     Input a query to search for papers on arXiv based on a topic and store their information.
#     Returns list of paper IDs found in the search.
#     """
#     # Lazy-load arxiv_search_chunk only when the tool is invoked
#     # tools_dir = Path(__file__).parent
#     # mod_path = tools_dir / "curaitor_agent/arxiv_search_chunk.py"
#     # spec = importlib.util.spec_from_file_location("arxiv_search_chunk", mod_path)
#     # mod = importlib.util.module_from_spec(spec)
#     # assert spec and spec.loader
#     # spec.loader.exec_module(mod)

#     # out_dir = save_dir or PAPER_DIR
#     # paper_ids = mod.download_arxiv_pdfs(query, output_dir=out_dir)
#     keywords = get_keywords_from_llm(query)
#     print(f"Extracted keywords: {keywords}")
#     return keywords

# def search_papers(topic: str, max_results: int = 5) -> List[str]:
#     """
#     Search for papers on arXiv based on a topic and store their information.

#     Args:
#         topic: The topic to search for
#         max_results: Maximum number of results to retrieve (default: 5)

#     Returns:
#         List of paper IDs found in the search
#     """
#
#     # Use arxiv to find the papers
#     client = arxiv.Client()
#
#     # Search for the most relevant articles matching the queried topic
#     search = arxiv.Search(
#         query = topic,
#         max_results = max_results,
#         sort_by = arxiv.SortCriterion.Relevance
#     )

#     papers = client.results(search)

#     # Create directory for this topic
#     path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
#     os.makedirs(path, exist_ok=True)

#     file_path = os.path.join(path, "papers_info.json")

#     # Try to load existing papers info
#     try:
#         with open(file_path, "r") as json_file:
#             papers_info = json.load(json_file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         papers_info = {}

#     # Process each paper and add to papers_info
#     paper_ids = []
#     for paper in papers:
#         paper_ids.append(paper.get_short_id())
#         paper_info = {
#             'title': paper.title,
#             'authors': [author.name for author in paper.authors],
#             'summary': paper.summary,
#             'pdf_url': paper.pdf_url,
#             'published': str(paper.published.date())
#         }
#         papers_info[paper.get_short_id()] = paper_info

#     # Save updated papers_info to json file
#     with open(file_path, "w") as json_file:
#         json.dump(papers_info, json_file, indent=2)
#
#     print(f"Results are saved in: {file_path}")
#
#     return paper_ids

# @mcp.tool()
# def extract_info(paper_id: str) -> str:
#     """
#     Search for information about a specific paper across all topic directories.

#     Args:
#         paper_id: The ID of the paper to look for
#
#     Returns:
#         JSON string with paper information if found, error message if not found
#     """

#     for item in os.listdir(PAPER_DIR):
#         item_path = os.path.join(PAPER_DIR, item)
#         if os.path.isdir(item_path):
#             file_path = os.path.join(item_path, "papers_info.json")
#             if os.path.isfile(file_path):
#                 try:
#                     with open(file_path, "r") as json_file:
#                         papers_info = json.load(json_file)
#                         if paper_id in papers_info:
#                             return json.dumps(papers_info[paper_id], indent=2)
#                 except (FileNotFoundError, json.JSONDecodeError) as e:
#                     print(f"Error reading {file_path}: {str(e)}")
#                     continue
#
#     return f"There's no saved information related to paper {paper_id}."
config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
# print(config['llm'][1]['model'])
provider = config['llm'][0]['provider']
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

def _normalize_model_for_provider(p: str, m: str | None) -> str | None:
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

@mcp.tool()
def get_keywords_from_llm(natural_language_query: str, model: str = None) -> list[str]:
    """
    Uses an LLM (OpenRouter | OpenAI | Google Gemini) to extract keywords and phrases from a natural language query.

    Args:
        natural_language_query: The user's research question or topic.
        model: The model name. For provider='openrouter', pass OpenRouter model id.
               For provider='openai' pass OpenAI model id (e.g., 'gpt-4o-mini').
               For provider='google' pass Gemini model id (e.g., 'gemini-1.5-flash' or 'gemini-2.0-flash-exp').

    Returns:
        A list of keywords and phrases suitable for an arXiv search.
    """
    # Ensure API key is set for the chosen provider
    env_hint = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider, "API_KEY")
    if not api_key:
        raise ValueError(f"No API key set for provider '{provider}'. Please export {env_hint}.")

    def _normalize_model_for_provider(p: str, m: str | None) -> str | None:
        if not m:
            return None
        m = m.strip()
        # If user accidentally passes OpenRouter-style names to direct providers, trim prefixes/suffixes
        if p in ("openai", "google"):
            if "/" in m:
                m = m.split("/")[-1]   # keep last segment
            if ":" in m:
                m = m.split(":")[0]    # drop qualifiers like ':free'
        return m

    model = _normalize_model_for_provider(provider, model)

    # A detailed prompt telling the LLM exactly what to do.
    prompt = f"""
    You are an expert academic researcher specializing in chemistry, biology, physics, computer science, and condensed matter physics.
    Your task is to analyze the following user query and extract a list of precise, effective keywords and multi-word phrases for searching the arXiv database.
    The goal is to find the most relevant academic papers.

    - Identify core concepts, technical terms, and important named entities.
    - For multi-word concepts (e.g., "machine learning potential"), keep them as a single phrase.
    - Do not include generic words.
    - Return ONLY a comma-separated list of these keywords and phrases. Do not add any explanation or introductory text.

    User Query: "{natural_language_query}"

    Keywords:
    """

    max_retries = 5
    backoff_base = 1.6

    def _sleep(attempt: int, retry_after: str | None):
        if retry_after and retry_after.isdigit():
            wait = float(retry_after)
        else:
            wait = backoff_base ** attempt
        # time_module.sleep(min(wait, 30))
        time_module.sleep(min(wait, 3))

    def _parse_keywords(text: str) -> list[str]:
        return [kw.strip() for kw in (text or "").split(",") if kw.strip()]

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    # optional but recommended by OpenRouter
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                llm_output = data["choices"][0]["message"]["content"]

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                llm_output = data["choices"][0]["message"]["content"]

            elif provider == "google":
                # Google Generative Language API (Gemini)
                g_model = model or "gemini-1.5-flash"
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {"temperature": 0},
                }
                response = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                # Join all text parts from the top candidate
                parts = data["candidates"][0]["content"]["parts"]
                llm_output = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))

            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            keywords = _parse_keywords(llm_output)
            print(f"[INFO] LLM ({provider}:{model}) extracted keywords: {keywords}")
            return keywords

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            body = e.response.text[:500] if e.response is not None else ""
            print(f"[ERROR] HTTP {status} from {provider}: {body}")
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                _sleep(attempt, e.response.headers.get("Retry-After") if e.response else None)
                continue
            return []
        except (KeyError, IndexError, ValueError) as e:
            print(f"[ERROR] Failed to parse {provider} response: {e}")
            try:
                print(f"Raw response: {response.text[:500]}")
            except Exception:
                pass
            return []
        except requests.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            if attempt < max_retries:
                _sleep(attempt, None)
                continue
            return []

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)