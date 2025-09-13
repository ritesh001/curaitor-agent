from rag.content_parsing import extract_pdf_components
import os
import json

class PDFProcessor:
    def __init__(self, output_path='data/processed/extracted_text.json'):
        self.output_path = output_path

    def process_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        
        # Extract components from the PDF
        extracted_data = extract_pdf_components(pdf_path)
        
        # Save extracted texts to output path
        self.save_extracted_data(extracted_data)

    def save_extracted_data(self, data):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Extracted data saved to {self.output_path}")