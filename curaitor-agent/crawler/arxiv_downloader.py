from pathlib import Path
import requests
import os

ARXIV_API_URL = "http://export.arxiv.org/api/query?"

def download_arxiv_pdfs(query: str, max_results: int = 5, output_dir: str = "data/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }

    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()

    entries = response.json().get('feed', {}).get('entry', [])
    for entry in entries:
        pdf_url = entry.get('link', [])[1].get('href')  # Get the PDF link
        title = entry.get('title', '').replace('/', '_').replace('\\', '_')  # Clean title for filename
        pdf_filename = f"{title}.pdf"
        pdf_path = Path(output_dir) / pdf_filename

        if not pdf_path.exists():
            pdf_response = requests.get(pdf_url)
            pdf_response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(pdf_response.content)
            print(f"Downloaded: {pdf_filename}")
        else:
            print(f"File already exists: {pdf_filename}")

# Example usage
if __name__ == "__main__":
    download_arxiv_pdfs("machine learning", max_results=5)