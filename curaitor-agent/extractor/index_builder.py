from rag.content_parsing import extract_pdf_components
import os
import json

class IndexBuilder:
    def __init__(self, output_path='data/index/index.json'):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def build_index(self, pdf_files):
        index = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                components = extract_pdf_components(pdf_file)
                index.append({
                    "file_path": pdf_file,
                    "texts": components['texts'],
                    "tables": components['tables'],
                    "images": components['images']
                })
            else:
                print(f"Warning: {pdf_file} does not exist.")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        print(f"Index built and saved to {self.output_path}")