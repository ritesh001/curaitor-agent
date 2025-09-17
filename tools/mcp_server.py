# This is a sample MCP server that provides tools to search for papers on arXiv
# and extract information about specific papers. It uses the `arxiv` library to
# Nanta, Shichuan
# Sep 2025
import arxiv
import json
import os
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# --- Notification imports ---
import importlib.util
notif_path = os.path.join(os.path.dirname(__file__), 'Notifications.Py')
spec = importlib.util.spec_from_file_location('Notifications', notif_path)
Notifications = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Notifications)
NotificationDispatcher = Notifications.NotificationDispatcher
NotificationRequest = Notifications.NotificationRequest

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
def send_notification(
    notification_types: List[str] | str,
    message: str,
    email: str = None,
    phone: str = None,
    slack_user: str = None,
    wechat_user: str = None
) -> Dict[str, Any]:
    """
    Send a notification using supported channels (email, whatsapp, wechat, slack).
    Args:
        notification_types: Channel(s) to use (str or list)
        message: Message to send
        email: Email address (if needed)
        phone: Phone number (if needed)
        slack_user: Slack user ID (if needed)
        wechat_user: WeChat user ID (if needed)
    Returns:
        Dict with status per channel
    """
    req = NotificationRequest(
        notification_types=notification_types,
        message=message,
        email=email,
        phone=phone,
        slack_user=slack_user,
        wechat_user=wechat_user
    )
    # Run the dispatcher (async)
    import asyncio
    result = asyncio.run(NotificationDispatcher.dispatch(req))
    return result


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)
