
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
def send_email_tool(params: dict) -> TextContent:
    """
    Send an email using Gmail. Supports plain text or HTML.

    Accepted call shapes (both valid):

      1) Top-level fields (preferred):
         {
           "to": "user@example.com",
           "subject": "Hello",
           "body": "Hi there",
           "html": false,
           "cc": "cc@example.com",          # optional
           "bcc": "hidden@example.com",     # optional
           "reply_to": "me@example.com",    # optional
           "from_alias": "Agent Bot"        # optional
         }

      2) Legacy wrapper:
         { "params": { ...same fields... } }

    Required: "to" (str), "subject" (str), "body" (str).
    Optional: "html" (bool), "cc", "bcc", "reply_to", "from_alias" (str).

    Success:
      { "success": true, "message_id": "...", "thread_id": "...", "to": "...", "subject": "...", "preview": "..." }

    Validation error:
      { "success": false, "error": { "code": "validation_error", ... } }

    Auth needed :
      { "success": false, "error": { "code": "auth_required", "message": "...", "auth_url": "https://accounts.google.com/..." } }
    """
    # ---- Normalize shape ----
    payload = params.get("params") if isinstance(params, dict) and "params" in params else params
    if not isinstance(payload, dict):
        example = {"to": "user@example.com", "subject": "Hello", "body": "Hi", "html": False}
        err = {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": "Top-level object must be a dict of fields OR contain a dict under key 'params'.",
                "expected_shape": "Provide fields at top-level OR under a top-level 'params' object.",
                "example": example,
            },
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    # ---- Required fields ----
    required = ["to", "subject", "body"]
    missing = [k for k in required if k not in payload]
    if missing:
        example = {"to": "user@example.com", "subject": "Hello", "body": "Hi", "html": False}
        err = {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": f"Missing required fields: {', '.join(missing)}",
                "expected_shape": "Provide fields at top-level OR under a top-level 'params' object.",
                "example": example,
            },
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    # ---- Type checks ----
    type_errors = []
    if not isinstance(payload.get("to"), str): type_errors.append("to must be a string")
    if not isinstance(payload.get("subject"), str): type_errors.append("subject must be a string")
    if not isinstance(payload.get("body"), str): type_errors.append("body must be a string")
    if "html" in payload and not isinstance(payload["html"], bool): type_errors.append("html must be a boolean")
    for opt in ["cc", "bcc", "reply_to", "from_alias"]:
        if opt in payload and not isinstance(payload[opt], str):
            type_errors.append(f"{opt} must be a string")
    if type_errors:
        err = {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": "; ".join(type_errors),
                "example": {"to":"user@example.com","subject":"Hello","body":"Hi","html":False},
            },
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    # ---- Call gmail_send ----
    allowed_keys = {"to", "subject", "body", "html", "cc", "bcc", "reply_to", "from_alias"}
    call_kwargs = {k: payload[k] for k in allowed_keys if k in payload}

    try:
        result = gmail_send(**call_kwargs)
        out = {
            "success": True,
            "message_id": result.get("id"),
            "thread_id": result.get("threadId"),
            "to": payload.get("to"),
            "subject": payload.get("subject"),
            "preview": (payload.get("body") or "")[:160],
        }
        return TextContent(type="text", text=json.dumps(out, indent=2))

    except AuthRequired as e:
        # New: Return a direct OAuth URL so the user can authenticate explicitly.
        err = {
            "success": False,
            "error": {
                "code": "auth_required",
                "message": (
                    "Authentication is required to send email. "
                    "Open the URL to grant access, then retry."
                ),
                "auth_url": e.auth_url,
                "details": e.original_message,  # e.g., "no method available for opening 'https:...'"
            },
            "input": {k: call_kwargs.get(k) for k in ("to", "subject", "html", "cc", "bcc", "reply_to", "from_alias")},
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    except Exception as e:
        # Generic failure
        err = {
            "success": False,
            "error": {
                "code": "send_failed",
                "message": str(e),
                "hint": "Check Gmail auth (credentials/token), scopes, and network. Ensure SCOPES includes gmail.send.",
            },
            "input": {k: call_kwargs.get(k) for k in ("to", "subject", "html", "cc", "bcc", "reply_to", "from_alias")},
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)