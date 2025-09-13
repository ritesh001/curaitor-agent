import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
CONFIG = BASE_DIR / "config.yaml"  # ensure your config lives here

def run_script(rel_path, *args):
    script = SCRIPTS_DIR / rel_path
    subprocess.run([sys.executable, str(script), *args], check=True)

def download_arxiv_pdfs(query: str):
    print(f"Downloading PDFs from arXiv for query: {query}")
    # pass --config and let downloader use LLM to rewrite the query
    run_script("download_arxiv.py", "--config", str(CONFIG), "--use-llm", query)

def extract_text_from_pdfs():
    print("Extracting text from downloaded PDFs...")
    run_script("extract_and_index.py")

def query_agent(user_query: str):
    print("Querying the AI agent...")
    run_script("ask.py", "--config", str(CONFIG), user_query)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your query (used for arXiv search and answering): ").strip()
        if not user_query:
            print("No query provided. Exiting.")
            sys.exit(1)

    download_arxiv_pdfs(user_query)
    extract_text_from_pdfs()
    query_agent(user_query)