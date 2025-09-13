from pdfminer.high_level import extract_text

def parse_pdf(file_path):
    text = extract_text(file_path)
    # simple split into chunks
    return text