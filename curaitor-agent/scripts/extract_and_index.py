from pathlib import Path
import json, argparse
from tqdm import tqdm
from rag.content_parsing import extract_pdf_components

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_ROOT = DATA_DIR / "raw"
PROC_ROOT = DATA_DIR / "processed"
INDEX_ROOT = DATA_DIR / "index"

def default_split(text: str, chunk_size=1500, overlap=150):
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def find_latest_manifest() -> Path | None:
    manifests = sorted(RAW_ROOT.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None

def extract_all(manifest_path: Path):
    items = json.loads(manifest_path.read_text())
    slug = Path(manifest_path).parent.name
    PROC_DIR = PROC_ROOT / slug
    INDEX_DIR = INDEX_ROOT / slug
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = INDEX_DIR / "corpus.jsonl"

    with open(corpus_path, "w", encoding="utf-8") as corpus_out:
        for item in tqdm(items, desc="Extracting PDFs"):
            pdf_path = Path(item["pdf_path"])
            if not pdf_path.exists():
                print(f"[WARN] Missing pdf {pdf_path}")
                continue
            try:
                result = extract_pdf_components(str(pdf_path))
            except Exception as e:
                print(f"[WARN] Docling failed on {pdf_path}: {e}")
                continue

            texts = result.get("texts", [])
            full_text = "\n".join(t.get("text","") for t in texts if isinstance(t, dict) and t.get("text"))
            if not full_text.strip():
                print(f"[WARN] No text extracted for {pdf_path}")
                continue

            doc_id = item["arxiv_id"]
            (PROC_DIR / f"{doc_id}.json").write_text(json.dumps({
                "doc_id": doc_id,
                "title": item["title"],
                "authors": item["authors"],
                "summary": item["summary"],
                "published": item["published"],
                "pdf_path": item["pdf_path"],
                "text_len": len(full_text),
            }, indent=2), encoding="utf-8")
            (PROC_DIR / f"{doc_id}.txt").write_text(full_text, encoding="utf-8")

            for i, chunk in enumerate(default_split(full_text)):
                corpus_out.write(json.dumps({
                    "doc_id": doc_id,
                    "title": item["title"],
                    "chunk_id": i,
                    "text": chunk,
                    "pdf_path": item["pdf_path"],
                }) + "\n")

    print(f"[INDEX] Wrote corpus: {corpus_path}")
    return corpus_path

def main():
    ap = argparse.ArgumentParser(description="Extract text and build per-query index")
    ap.add_argument("--manifest", type=Path, default=None, help="path to manifest.json")
    args = ap.parse_args()

    manifest = args.manifest or find_latest_manifest()
    if not manifest or not manifest.exists():
        print(f"No manifest found. Run download_arxiv.py with a query first.")
        return
    extract_all(manifest)

if __name__ == "__main__":
    main()