
# This is a sample MCP server that provides tools to search for papers on arXiv
# and extract information about specific papers. It uses the `arxiv` library to
# Nanta, Shichuan
# Sep 2025
import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from arxiv_search_chunk import download_arxiv_pdfs
import yaml
import subprocess
import numpy as np
from pathlib import Path
import importlib.util
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

@mcp.tool()
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
def input_query(query: str, max_days: int = 30, save_dir: str = None) -> List[str]:
    """
    Input a query to search for papers on arXiv based on a topic and store their information.
    Returns list of paper IDs found in the search.
    """
    # Lazy-load arxiv_search_chunk only when the tool is invoked
    tools_dir = Path(__file__).parent
    mod_path = tools_dir / "tools/arxiv_search_chunk.py"
    spec = importlib.util.spec_from_file_location("arxiv_search_chunk", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    out_dir = save_dir or PAPER_DIR
    paper_ids = mod.download_arxiv_pdfs(query, output_dir=out_dir)
    return paper_ids

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



if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)