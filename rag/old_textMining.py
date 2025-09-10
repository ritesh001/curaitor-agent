"""
File name: textMining.py
Author: Stephanie Khuu
Date created: 2024-04-16
Date last modified: 2024-05-24
Python Version: 3.11.4

Description:
This script allows for text mining of pdfs and storage as .txt file
"""

from io import StringIO
import os, sys
import re


# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfdocument import PDFDocument
# from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
# from pdfminer.pdfpage import PDFPage
# from pdfminer.pdfparser import PDFParser
import docling
from tqdm import tqdm

from pathlib import Path
# from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
# from docling.document import SimpleDocxExporter  #


# def clean_pdf_text(text):
#     # Define patterns to identify the Abstract and References sections
#     abstract_pattern = r"ABSTRACT:.*?(?=\n[A-Z]{2,}[ \n])"
#     headings_pattern = r"^\b(METHODS|MATERIALS AND METHODS|INTRODUCTION|BACKGROUND|RESULTS|DISCUSSION|CONCLUSIONS|INVITED REVIEW)\b(?=\n)"
#     references_pattern = r"REFERENCES\n.*"
#     acknowledgement_pattern = r"Acknowledgments.*"
#     # unwanted_lines_pattern = r"^.+?(\bVol\.\n|www\.|Copyright|http|\bVol\b|downloaded).*$"

#     # Remove unwanted lines
#     # text = re.sub(unwanted_lines_pattern, '', text, flags=re.MULTILINE|re.IGNORECASE)

#     # Remove the identified sections from the text
#     # First remove the references
#     clean_text_references = re.sub(
#         references_pattern, "", text, flags=re.IGNORECASE
#     )
#     clean_text_references = re.sub(
#         r"(\d+\.|[A-Z]+\d+)\s.*?(?=\n\n)",
#         "",
#         clean_text_references,
#         flags=re.IGNORECASE,
#     )

#     # Then remove acknowledgements and below
#     clean_text_acknowledgements = re.sub(
#         acknowledgement_pattern, "", clean_text_references, flags=re.IGNORECASE
#     )

#     # Then remove the abstract
#     clean_text_abstract = re.sub(
#         abstract_pattern, "", clean_text_acknowledgements, flags=re.IGNORECASE
#     )

#     # Then remove the headings
#     clean_text_methods = re.sub(
#         headings_pattern, "", clean_text_abstract, flags=re.IGNORECASE
#     )

#     # Limit search for 'accepted' or 'introduction' within the first 200 words
#     first_200_words = " ".join(clean_text_methods.split()[:200])

#     # Remove all text (including titles) above accepted publication date
#     accepted_match = re.search(
#         r"accepted", first_200_words, flags=re.IGNORECASE
#     )
#     intro_match = re.search(r"intro", first_200_words, flags=re.IGNORECASE)

#     if accepted_match:
#         start_line = (
#             clean_text_methods.rfind("\n", 0, accepted_match.start()) + 1
#         )
#         # Remove all text preceding the first occurrence of "accepted"
#         clean_text = clean_text_methods[start_line:]
#     elif intro_match:
#         start_line = clean_text_methods.rfind("\n", 0, intro_match.start()) + 1
#         # Remove all text preceding the first occurrence of "accepted"
#         clean_text = clean_text_methods[start_line:]
#     else:
#         clean_text = clean_text_methods

#     return clean_text


# def read_pdf(pdf_path):
#     # Read the PDF and extract its text

#     # Open the PDF file in binary mode
#     with open(pdf_path, "rb") as file:
#         # Create a StringIO object to hold the extracted text
#         output_string = StringIO()

#         # Create a PDFParser object to parse the PDF file
#         parser = PDFParser(file)

#         # Create a PDFDocument object to represent the parsed PDF
#         doc = PDFDocument(parser)

#         # Create a PDFResourceManager object to manage shared resources
#         rsrcmgr = PDFResourceManager()

#         # Create a TextConverter object to convert PDF content to text
#         device = TextConverter(rsrcmgr, output_string, laparams=LAParams())

#         # Create a PDFPageInterpreter object to process PDF pages
#         interpreter = PDFPageInterpreter(rsrcmgr, device)

#         # Iterate over each page in the PDF
#         for page in PDFPage.create_pages(doc):
#             # Process the page using the interpreter and convert content to text
#             interpreter.process_page(page)

#         # Get the extracted text from the StringIO object
#         text = output_string.getvalue()

#         # Close the StringIO object
#         output_string.close()

#     # Return the extracted text
#     return text


# # def extract_tables_from_pdf(pdf_path):
# #     table_pattern = r"(?i)(?:table\.?)\s*\d+[\s\S]{0,100}?(?:\n.+){0,2}"
# #     tabledfs = tabula.read_pdf(pdf_path, pages ='all', multiple_tables=True)
# #     valid_tables = []

# #     # Filter tables using the defined pattern
# #     for table in tabledfs:
# #         if any(re.search(table_pattern,col, re.IGNORECASE) for col in table.columns) or re.search(table_pattern, table.to_string()):
# #             try:
# #                 # Convert all columns of the table to numeric types, handle errors
# #                 table = table.apply(pd.to_numeric, errors='coerce')
# #                 valid_tables.append(table)
# #             except Exception as e:
# #                 print(f"Error processing table: {e}")

# #     return valid_tables


# def write_cleaned_text(cleaned_text, output_path):
#     # Write the cleaned text to an output file
#     with open(output_path, "w", encoding="utf-8") as file:
#         file.write(cleaned_text)


# def process_pdf(input_pdf_path, output_directory):
#     # Read the PDF
#     text = read_pdf(input_pdf_path)

#     # Construct output path
#     output_filename = (
#         os.path.splitext(os.path.basename(input_pdf_path))[0]
#         + "_output_text.txt"
#     )
#     output_path = os.path.join(output_directory, output_filename)

#     # Clean the text
#     cleaned_text = clean_pdf_text(text)

#     # tabledfs = extract_tables_from_pdf(input_pdf_path)
#     # with open(output_tablePath, 'w') as f:
#     #     for index, table in enumerate(tabledfs, start=1):
#     #         f.write(f"Table{index + 1}: \n")
#     #         table.to_csv(f, index=False, header=True, sep='\t')  # Use tab to separate columns
#     #         f.write("\n\n")
#     #         print(f"Table {index}:")
#     #         print(table)
#     # print("Tables have been saved to the text file.")

#     # Write cleaned text to output file
#     write_cleaned_text(cleaned_text, output_path)
#     print(f"Cleaned text has been written to file: {output_path}")


# def process_all_pdfs(pdfs_directory, output_directory):
#     # Ensure the output directory exists, create it if not
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory, exist_ok=True)

#     # List all PDF files in the directory
#     pdf_files = [f for f in os.listdir(pdfs_directory) if f.endswith(".pdf")]

#     # Iterate through PDF files in the directory
#     for filename in tqdm(pdf_files, desc="Mining PDFs"):
#         print(filename)
#         input_pdf_path = os.path.join(pdfs_directory, filename)
#         process_pdf(input_pdf_path, output_directory)


# def clean_text_file(file_path):
#     # Define the pattern for lines to remove
#     unwanted_lines_pattern = re.compile(
#         r"^.+?(published|doi|\bpublication\b|\barticle\b|\bcorrespondence\b|\bDept\. \b|E-mail|Downloaded|\bVol\.\b|www\.|Copyright|http|journals|INVITED REVIEW|Department|University|Univ\.).*$",
#         re.MULTILINE | re.IGNORECASE,
#     )
#     single_character_line_pattern = re.compile(r"^\s*[a-zA-Z]\s*$")
#     number_line_pattern = re.compile(
#         r"^\s*\d+\s*$"
#     )  # Pattern to match lines with only numbers
#     blank_line_pattern = re.compile(r"^\s*$")  # Pattern to match blank lines

#     # Read the content of the file
#     with open(file_path, "r", encoding="utf-8") as file:
#         text = file.read()

#     # Filter out unwanted lines using unwanted_lines_pattern
#     cleaned_text = unwanted_lines_pattern.sub("", text)

#     # Split into lines, filter out single character lines, and join back into a single string
#     cleaned_lines = [
#         line
#         for line in cleaned_text.splitlines()
#         if not single_character_line_pattern.match(line)
#         and not number_line_pattern.match(line)
#         and not blank_line_pattern.match(line)
#     ]
#     cleaned_text = "\n".join(cleaned_lines)

#     # Write the cleaned lines back to the file
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(cleaned_text)


# def process_text_files(directory):
#     # Iterate through all text files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(directory, filename)
#             print(f"Cleaning {filename}...")
#             clean_text_file(file_path)
#             print(f"{filename} has been cleaned.")


# # # Directory containing your text files
# # text_files_directory = 'articles/experimental/output/textOutput'

# # # Directory containing PDFs to process
# # pdfs_directory = 'articles/experimental/output'

# # # Output directory for cleaned text files
# # output_directory = os.path.join(pdfs_directory, 'textOutput')

# # # Process all PDFs in the directory
# # process_all_pdfs(pdfs_directory, output_directory)
# # process_text_files(text_files_directory) optional exporters

pdf_path = sys.argv[1]  # Path to the PDF file

# pipeline = StandardPdfPipeline()         # Handles OCR fallback, layout, tables, figures

# try:
#     # Newer API
#     from docling.pipeline.standard_pdf_pipeline import (
#         StandardPdfPipeline,
#         StandardPdfPipelineOptions,
#     )
# except ImportError:
#     print("older docling API...")
    # Older API variants
    # try:
    #     from docling.pipeline.standard_pdf import (
    #         StandardPdfPipeline,
    #         StandardPdfPipelineOptions,
    #     )
    # except ImportError:
    #     from docling.pipeline.standard import StandardPdfPipeline
    #     StandardPdfPipelineOptions = None
# ...existing code...

# Replace the single line creating the pipeline with the block below:

# Always define (avoid NameError)
StandardPdfPipelineOptions = None
pipeline_cls = None

# Import attempts (newest -> older)
try:
    from docling.pipeline.standard_pdf_pipeline import (
        StandardPdfPipeline as pipeline_cls,
        StandardPdfPipelineOptions,
    )
except ImportError:
    try:
        from docling.pipeline.standard_pdf import (
            StandardPdfPipeline as pipeline_cls,
            StandardPdfPipelineOptions,
        )
    except ImportError:
        try:
            from docling.pipeline.standard import StandardPdfPipeline as pipeline_cls
        except ImportError as e:
            raise SystemExit(
                "Cannot import Docling pipeline. Install/upgrade:\n"
                "  pip install -U \"docling[pdf]\""
            ) from e

# result = pipeline.run(pdf_path)

# doc = result.document   # Structured Document object

# # 1. Extract full plain text (ordered by reading sequence)
# full_text = doc.export_to_text()

# # 2. Extract structured sections (headings + paragraphs)
# sections = []
# for sec in doc.sections:
#     sections.append({
#         "heading": sec.title.text if sec.title else None,
#         "level": sec.level if hasattr(sec, "level") else None,
#         "text": "\n".join(p.text for p in sec.paragraphs)
#     })

# # 3. Tables
# tables_data = []
# for i, t in enumerate(doc.tables):
#     # Raw cells
#     cells = [[cell.text for cell in row.cells] for row in t.rows]
#     tables_data.append({
#         "index": i,
#         "title": t.title.text if t.title else None,
#         "n_rows": len(t.rows),
#         "n_cols": len(t.rows[0].cells) if t.rows else 0,
#         "cells": cells
#     })
#     # Optional: export as CSV
#     csv_path = Path(f"table_{i}.csv")
#     with csv_path.open("w", encoding="utf-8") as f:
#         for row in cells:
#             f.write(",".join(r.replace(",", " ")) + "\n")

# # 4. Images (figures)
# images_info = []
# for i, fig in enumerate(doc.figures):
#     img = fig.image
#     out_path = Path(f"figure_{i}.{img.ext or 'png'}")
#     out_path.write_bytes(img.bytes)
#     images_info.append({
#         "index": i,
#         "title": fig.caption.text if fig.caption else None,
#         "path": str(out_path),
#         "width": img.width,
#         "height": img.height,
#         "dpi": img.dpi
#     })

# # 5. (Optional) Export to DOCX / JSON
# # docx_bytes = SimpleDocxExporter().export(doc)
# # Path("output.docx").write_bytes(docx_bytes)

# # Example aggregate result
# extracted = {
#     "full_text": full_text,
#     "sections": sections,
#     "tables": tables_data,
#     "figures": images_info
# }

def build_pipeline():
    # Newer API requires options; older ignores / errors
    if StandardPdfPipelineOptions is not None:
        try:
            opts = StandardPdfPipelineOptions(
                # Uncomment / adjust:
                # enable_ocr=True,
                # extract_tables=True,
                # extract_figures=True,
            )
            return pipeline_cls(pipeline_options=opts)
        except TypeError:
            # Signature mismatch, try bare
            return pipeline_cls()
    return pipeline_cls()

def extract(pdf_path: Path):
    pipeline = build_pipeline()
    result = pipeline.run(str(pdf_path))
    doc = result.document

    # 1. Full text
    full_text = doc.export_to_text()

    # 2. Sections
    sections = []
    for sec in getattr(doc, "sections", []):
        sections.append({
            "heading": getattr(getattr(sec, "title", None), "text", None),
            "level": getattr(sec, "level", None),
            "text": "\n".join(p.text for p in getattr(sec, "paragraphs", []))
        })

    # 3. Tables
    tables_data = []
    for i, t in enumerate(getattr(doc, "tables", [])):
        rows = []
        for row in t.rows:
            rows.append([cell.text for cell in row.cells])
        tables_data.append({
            "index": i,
            "title": getattr(getattr(t, "title", None), "text", None),
            "n_rows": len(rows),
            "n_cols": len(rows[0]) if rows else 0,
            "cells": rows
        })

    # 4. Figures / images
    figures = []
    for i, fig in enumerate(getattr(doc, "figures", [])):
        img = fig.image
        figures.append({
            "index": i,
            "caption": getattr(getattr(fig, "caption", None), "text", None),
            "format": img.ext or "png",
            "width": getattr(img, "width", None),
            "height": getattr(img, "height", None),
            "dpi": getattr(img, "dpi", None),
            "bytes_len": len(img.bytes) if getattr(img, "bytes", None) else 0,
        })

    return {
        "full_text": full_text,
        "sections": sections,
        "tables": tables_data,
        "figures": figures
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python textMining.py <pdf_path> [--dump-json out.json] [--dump-assets dir]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    dump_json = None
    dump_assets = None
    if "--dump-json" in sys.argv:
        idx = sys.argv.index("--dump-json")
        if idx + 1 < len(sys.argv):
            dump_json = Path(sys.argv[idx + 1])
    if "--dump-assets" in sys.argv:
        idx = sys.argv.index("--dump-assets")
        if idx + 1 < len(sys.argv):
            dump_assets = Path(sys.argv[idx + 1])
            dump_assets.mkdir(parents=True, exist_ok=True)

    data = extract(pdf_path)

    # Write assets (tables / figures) if requested
    if dump_assets:
        # Full text
        (dump_assets / f"{pdf_path.stem}_text.txt").write_text(
            data["full_text"], encoding="utf-8"
        )
        # Tables as CSV
        for t in data["tables"]:
            with (dump_assets / f"table_{t['index']}.csv").open("w", encoding="utf-8") as f:
                for row in t["cells"]:
                    f.write(",".join(c.replace(",", " ")) + "\n")
        # Figures binary
        # Need to re-run minimal figure export to get bytes
        pipeline = build_pipeline()
        doc = pipeline.run(str(pdf_path)).document
        figs_dir = dump_assets / "figures"
        figs_dir.mkdir(exist_ok=True)
        for i, fig in enumerate(getattr(doc, "figures", [])):
            img = fig.image
            ext = img.ext or "png"
            (figs_dir / f"figure_{i}.{ext}").write_bytes(img.bytes)
            if fig.caption:
                (figs_dir / f"figure_{i}_caption.txt").write_text(fig.caption.text, encoding="utf-8")

    if dump_json:
        dump_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # Print brief summary
        print(f"Pages text length: {len(data['full_text'])}")
        print(f"Sections: {len(data['sections'])}")
        print(f"Tables: {len(data['tables'])}")
        print(f"Figures: {len(data['figures'])}")

if __name__ == "__main__":
    main()