
# This is a sample MCP server that provides tools to search for papers on arXiv
# and extract information about specific papers. It uses the `arxiv` library to

import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from mcp.server.fastmcp.prompts.base import Message
from gmail_send import gmail_send


# from pathlib import Path
# import argparse

PAPER_DIR = "data/agent_papers"

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

@mcp.tool()
# async def send_email_tool(params: dict) -> TextContent:
def send_email_tool(params: dict) -> TextContent:
    """
        Send an email using Gmail.
        Supports plain text or HTML.
        format of params (example in note.txt):
        {
            "to": {"type":"string", "description":"Recipient email address", default: "nsophonrat2@gmail.com"},
            "subject": {"type":"string"},
            "body": {"type":"string", "description":"Message body (plain text by default)"},
            "html": {"type":"boolean", "default": False},
            },
            "required": ["to","subject","body"],
        }
    )
    """
    required_fields = ["to", "subject", "body"]
    missing = [field for field in required_fields if not params.get(field)]
    if missing:
        error_msg = f"Missing required fields: {', '.join(missing)}"
        # return Message(role="tool", content=[TextContent(text=json.dumps({"error": error_msg}, indent=2))])
        # return Message(role="tool", content=[{"type": "text", "text": json.dumps(result, indent=2)}])
        return TextContent(type="text", text=json.dumps(result, indent=2))
    result = gmail_send(**params)
    print(f"Email sent result: {result}")
    # return Message(role="tool", content=[TextContent(text=json.dumps(result, indent=2))])
    # return Message(role="tool", content=[{"type": "text", "text": json.dumps(result, indent=2)}])
    # return result
    return TextContent(type="text", text=json.dumps(result, indent=2))

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)