from __future__ import annotations

import io
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os

import pandas as pd

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import (
    DoclingDocument,
    TextItem,
    SectionHeaderItem,
    PictureItem,
    TableItem,
)

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def build_converter(
    *,
    ocr: bool = False,
    image_scale: float = 2.0,
    table_mode: str = "ACCURATE",  # or "FAST"
    enable_table_images: bool = True,
) -> DocumentConverter:
    """
    Configure the PDF pipeline to keep page images (for cropping figures/tables),
    generate figure/table crops, and build structured tables.
    """
    p = PdfPipelineOptions()
    p.do_ocr = ocr
    p.do_table_structure = True
    p.table_structure_options.mode = getattr(TableFormerMode, table_mode)
    p.generate_page_images = True
    p.generate_picture_images = True
    p.generate_table_images = enable_table_images
    p.images_scale = image_scale

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=p)}
    )

def _page_no(item) -> Optional[int]:
    return item.prov[0].page_no if getattr(item, "prov", None) else None

def _bbox(item) -> Optional[Tuple[float, float, float, float]]:
    return tuple(item.bbox.as_tuple()) if getattr(item, "bbox", None) else None

def _ref(item) -> Optional[str]:
    return item.get_ref() if hasattr(item, "get_ref") else getattr(item, "self_ref", None)

def extract_pdf_components(
    pdf: str | Path,
    converter: Optional[DocumentConverter] = None,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - 'texts': List[dict]  (headings/paragraphs/etc.)
      - 'tables': List[dict] (df, caption, bytes/png, metadata)
      - 'images': List[dict] (bytes/png, caption, metadata)
      - 'doc': DoclingDocument (if you want to do advanced ops later)
    """
    if converter is None:
        converter = build_converter()

    conv = converter.convert(str(pdf))
    doc: DoclingDocument = conv.document

    # -------- TEXT (reading order) --------
    texts: List[Dict[str, Any]] = []
    order = 0
    for item, _level in doc.iterate_items(with_groups=False):
        if isinstance(item, (TextItem, SectionHeaderItem)):
            text = getattr(item, "text", None)
            if not text:
                continue
            typ = "heading" if isinstance(item, SectionHeaderItem) else "text"
            texts.append(
                {
                    "ref": _ref(item),
                    "type": typ,
                    "level": getattr(item, "level", None),
                    "text": text,
                    "page": _page_no(item),
                    "bbox": _bbox(item),
                    "order": order,
                }
            )
            order += 1

    # -------- TABLES (structured) --------
    tables: List[Dict[str, Any]] = []
    for t in doc.tables:
        df: pd.DataFrame = t.export_to_dataframe(doc)
        caption = t.caption_text(doc)

        img_bytes = None
        try:
            pil = t.get_image(doc)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception:
            pass

        tables.append(
            {
                "ref": _ref(t),
                "page": _page_no(t),
                "bbox": _bbox(t),
                "caption": caption or None,
                "dataframe": df,
                "image_png": img_bytes,
                "image_sha256": _hash_bytes(img_bytes) if img_bytes else None,
            }
        )

    # -------- IMAGES (figures) --------
    images: List[Dict[str, Any]] = []
    for p in doc.pictures:
        caption = p.caption_text(doc)
        img_bytes = None
        try:
            pil = p.get_image(doc)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception:
            pass
        images.append(
            {
                "ref": _ref(p),
                "page": _page_no(p),
                "bbox": _bbox(p),
                "caption": caption or None,
                "image_png": img_bytes,
                "image_sha256": _hash_bytes(img_bytes) if img_bytes else None,
            }
        )

    return {"texts": texts, "tables": tables, "images": images, "doc": doc}

# -------- Example usage --------
if __name__ == "__main__":
    print("Extracting components from PDF...")
    converter = build_converter(ocr=False, image_scale=2.0, table_mode="ACCURATE")
    result = extract_pdf_components(sys.argv[1], converter)
    print("Done.")
    texts = result["texts"]
    tables = result["tables"]
    images = result["images"]

    # Example: convert table DataFrames to records for vector stores
    table_records = [
        {
            "ref": t["ref"],
            "page": t["page"],
            "caption": t["caption"],
            "rows": t["dataframe"].to_dict(orient="records"),
        }
        for t in tables
    ]