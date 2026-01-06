#!/usr/bin/env python
"""
Run the LangGraph pipeline and persist results to a SQLite database.

Example:
  uv run python scripts/run_daily.py \
    --query "plastic recycling" \
    --max-days 7 \
    --db data/curaitor.sqlite
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curaitor_agent.langraph_pipeline import run_pipeline
from curaitor_agent.db_utils import init_db, upsert_docs, upsert_hits, insert_answer


def _write_search_results_csv(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arxiv_id",
                "title",
                "published",
                "updated",
                "entry_id",
                "pdf_url",
                "categories",
                "authors",
                "summary",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "arxiv_id": r.entry_id.split("/")[-1] if getattr(r, "entry_id", None) else "",
                    "title": getattr(r, "title", ""),
                    "published": (r.published.isoformat() if getattr(r, "published", None) else ""),
                    "updated": (r.updated.isoformat() if getattr(r, "updated", None) else ""),
                    "entry_id": getattr(r, "entry_id", ""),
                    "pdf_url": (r.pdf_url if getattr(r, "pdf_url", None) else ""),
                    "categories": " ".join(getattr(r, "categories", []) or []),
                    "authors": "; ".join([a.name for a in getattr(r, "authors", []) or []]),
                    "summary": " ".join((getattr(r, "summary", "") or "").split()),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Run LangGraph pipeline and update DB at regular intervals")
    parser.add_argument("--query", type=str, required=True, help="Natural language research query")
    parser.add_argument("--max-days", type=int, default=7, help="Search window in days")
    parser.add_argument("--db", type=str, default="data/curaitor.sqlite", help="SQLite DB path")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable embeddings/RAG retrieval")
    parser.add_argument("--faiss-per-query", type=int, default=None)
    parser.add_argument("--final-docs", type=int, default=None)
    parser.add_argument("--per-doc-chunks", type=int, default=None)
    parser.add_argument("--recency-alpha", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--save-npz", type=str, default=None)
    args = parser.parse_args()

    db_path = args.db
    init_db(db_path)

    state = run_pipeline(
        query=args.query,
        max_days=args.max_days,
        build_embeddings=not args.no_embeddings,
        faiss_per_query=args.faiss_per_query,
        final_docs=args.final_docs,
        per_doc_chunks=args.per_doc_chunks,
        recency_alpha=args.recency_alpha,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        save_npz=args.save_npz,
    )

    results = state.get("results", []) or []
    _write_search_results_csv(Path("search_results.csv"), results)

    docs = state.get("docs", []) or []
    hits = state.get("hits", []) or []
    final_answer = state.get("final_answer", "") or ""

    n_docs = upsert_docs(db_path, docs)
    n_hits = upsert_hits(db_path, hits)
    ans_id = insert_answer(db_path, args.query, final_answer) if final_answer else None

    print({
        "db": db_path,
        "docs_upserted": n_docs,
        "hits_upserted": n_hits,
        "answer_id": ans_id,
    })


if __name__ == "__main__":
    main()
