from crawler.arxiv_downloader import download_arxiv_pdfs
from rag.content_parsing import extract_pdf_components
from agent.llm_agent import ExtractionAgent
import os
import json

def run_crawler():
    # Step 1: Download PDFs from arXiv
    print("Downloading PDFs from arXiv...")
    download_arxiv_pdfs()
    
    # Step 2: Extract text from downloaded PDFs
    pdf_directory = 'data/raw'  # Assuming PDFs are stored here
    extracted_data = []
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Extracting components from {pdf_path}...")
            components = extract_pdf_components(pdf_path)
            extracted_data.append({
                "file_name": filename,
                "texts": components['texts'],
                "tables": components['tables'],
                "images": components['images']
            })
    
    # Step 3: Save extracted data to a JSON file
    output_path = 'data/processed/extracted_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2)
    
    print(f"Extraction complete. Results saved to {output_path}")