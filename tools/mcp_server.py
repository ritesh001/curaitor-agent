import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP

# from pathlib import Path
# import argparse

PAPER_DIR = "papers"


# # Initialize FastMCP server
# def parse_args():
#     """Parse command line arguments for MCP server."""
#     parser = argparse.ArgumentParser(description="Paper Search MCP Server")
#     parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
#     parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
#     parser.add_argument('--log-level', default='INFO', 
#                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
#                        help='Logging level (default: INFO)')
#     try:
#         args = parser.parse_args()
#     except SystemExit:
#         class Args:
#             port = 50001
#             host = '0.0.0.0'
#             log_level = 'INFO'
#         args = Args()
#     return args

# args = parse_args()
# mcp = FastMCP("paper_search", port=args.port, host=args.host)

mcp = FastMCP("paper_search")

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."



if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)