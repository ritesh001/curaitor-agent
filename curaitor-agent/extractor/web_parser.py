from rag.content_parsing import extract_pdf_components
from crawler.arxiv_downloader import download_arxiv_pdfs
from agent.llm_agent import ExtractionAgent
import json
import os

class WebParser:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.agent = ExtractionAgent(self.config)

    def download_and_extract(self, arxiv_ids):
        # Download PDFs from arXiv
        pdf_paths = download_arxiv_pdfs(arxiv_ids)
        all_results = []

        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                # Extract text components from the PDF
                extracted_data = extract_pdf_components(pdf_path)
                texts = extracted_data['texts']
                
                # Process each text component
                for text_item in texts:
                    text = text_item['text']
                    if text.strip():
                        res = self.agent.extract(text, self.config['extraction']['schema'])
                        all_results.append({
                            "source": pdf_path,
                            "extraction": res
                        })

        return all_results

    def save_results(self, results, output_path='data/output.json'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {output_path}")