from __future__ import annotations

"""
LangGraph-based orchestration for the literature RAG pipeline.

This graph reuses existing functions in curaitor_agent/arxiv_search_chunk.py and
content parsing helpers to perform:

1) Keyword extraction via LLM
2) arXiv search + PDF download (time-windowed)
3) PDF parsing to normalized docs
4) Token-based chunking
5) Embeddings + FAISS + rerank + recency scoring
6) Context assembly + final answer
7) Optional NPZ artifact saving

Usage:
  uv run python -m curaitor_agent.langraph_pipeline --query "your question"

Configuration is read from config.yaml and env vars, same as the existing pipeline.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime, time, timedelta
from dateutil import tz
from pathlib import Path
import os
import yaml
import numpy as np
import arxiv

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv, find_dotenv

# Load environment variables from a local .env if present (project root or parents)
try:
    # Try standard discovery first
    found = find_dotenv()
    if found:
        load_dotenv(found, override=False)
    else:
        # Fallback: load .env from repo root relative to this file
        repo_root = Path(__file__).resolve().parents[1]
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(env_path.as_posix(), override=False)
except Exception:
    pass

# Prefer package-relative imports; fall back to absolute if run as a script path.
try:
    from .content_parsing import extract_pdf_components, texts_to_plaintext
    from .arxiv_search_chunk import (
        get_keywords_from_llm,
        format_arxiv_query,
        download_arxiv_pdfs,
        chunk_text,
        _recency_score,
        format_context,
        pretty_print_docs,
        answer_query_with_context,
        LLM_MODEL,
    )
except Exception:  # pragma: no cover - fallback for direct script execution
    from curaitor_agent.content_parsing import extract_pdf_components, texts_to_plaintext
    from curaitor_agent.arxiv_search_chunk import (
        get_keywords_from_llm,
        format_arxiv_query,
        download_arxiv_pdfs,
        chunk_text,
        _recency_score,
        format_context,
        pretty_print_docs,
        answer_query_with_context,
        LLM_MODEL,
    )
    # get_keywords_from_llm,
    # format_arxiv_query,
    # download_arxiv_pdfs,
    # chunk_text,
    # _recency_score,
    # format_context,
    # pretty_print_docs,
    # answer_query_with_context,
    # LLM_MODEL,
    # )


# Load shared config
CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))


class PipelineState(TypedDict, total=False):
    # Inputs
    query: str
    max_days: int
    build_embeddings: bool
    faiss_per_query: int
    final_docs: int
    per_doc_chunks: int
    recency_alpha: float
    chunk_size: int
    overlap: int
    save_npz: Optional[str]

    # Derived
    keywords: List[str]
    arxiv_query: str

    # arXiv results and normalized docs
    results: List[Any]
    docs: List[Dict[str, Any]]

    # Chunking and retrieval
    chunks: List[Dict[str, Any]]
    hits: List[Dict[str, Any]]

    # Answering
    context: str
    final_answer: str


def _init_defaults(state: PipelineState) -> PipelineState:
    # Fill defaults from config if not provided
    if "query" not in state or not state["query"]:
        q = CONFIG["input"][0]["query"]
        state["query"] = f"search_query: {q}"  # compatible with nomic embed prompt format
    state.setdefault("max_days", CONFIG["input"][1]["max_days"])  # e.g., 1 or 2
    state.setdefault("build_embeddings", True)
    # Defaults aligned with main() in arxiv_search_chunk.py
    state.setdefault("faiss_per_query", 80)
    state.setdefault("final_docs", 12)
    state.setdefault("per_doc_chunks", CONFIG["rag"][2]["per_doc_chunks"])  # from config
    state.setdefault("recency_alpha", 0.35)
    state.setdefault("chunk_size", CONFIG["rag"][4]["chunk_size"])  # from config
    state.setdefault("overlap", CONFIG["rag"][5]["overlap"])        # from config
    state.setdefault("save_npz", CONFIG["output"][2]["npz_path"])             # arxiv_out_hits.npz
    return state


def node_keywords(state: PipelineState) -> PipelineState:
    query = state["query"]
    keywords = get_keywords_from_llm(query, model=LLM_MODEL)
    arxiv_query = format_arxiv_query(keywords)
    return {**state, "keywords": keywords, "arxiv_query": arxiv_query}


def node_arxiv_search_and_download(state: PipelineState) -> PipelineState:
    if not state.get("arxiv_query"):
        # Nothing to search
        return state

    tz_london = tz.gettz("Europe/London")
    now_local = datetime.now(tz_london)
    max_days = int(state["max_days"]) if state.get("max_days") else 1
    start_date = (now_local.date() - timedelta(days=max_days - 1))
    start_dt = datetime.combine(start_date, time(0, 0, tzinfo=tz_london))
    end_dt = datetime.combine(now_local.date(), time(23, 59, 59, tzinfo=tz_london))

    def _in_window(dt_utc):
        dt_local = dt_utc.astimezone(tz_london)
        return start_dt <= dt_local <= end_dt

    client = arxiv.Client(page_size=100, delay_seconds=3)
    search = arxiv.Search(
        query=state["arxiv_query"],
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results: List[Any] = []
    for r in client.results(search):
        ts = r.published or r.updated
        if ts and _in_window(ts):
            results.append(r)
        else:
            if ts is not None:
                ts_local = ts.astimezone(tz_london)
                if ts_local < start_dt:
                    break

    # Download PDFs into configured folder
    out_dir = CONFIG["source"][0]["pdf_path"]
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
        try:
            download_arxiv_pdfs(pdf_url=pdf_url, output_dir=out_dir)
        except Exception:
            # continue on best-effort
            pass

    return {**state, "results": results}


def node_build_docs(state: PipelineState) -> PipelineState:
    results = state.get("results", [])
    docs: List[Dict[str, Any]] = []
    pdf_base = CONFIG["source"][0]["pdf_path"]
    for r in results:
        arxiv_id = r.entry_id.split("/")[-1]
        pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
        title_abs = f"{r.title}\n\n{(' '.join(r.summary.split())).strip()}"
        pdf_filename = pdf_url.split("/")[-1] + ".pdf"
        pdf_path = Path(pdf_base) / pdf_filename

        try:
            comps = extract_pdf_components(pdf_path)
            body_text = texts_to_plaintext(comps["texts"])  # normalized plain text
        except Exception:
            body_text = ""

        full_text = ("search_document: " + title_abs + ("\n\n" + body_text if body_text else "")).strip()

        docs.append(
            {
                "doc_id": arxiv_id,
                "title": r.title,
                "text": full_text,
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "entry_id": r.entry_id,
                    "pdf_url": pdf_url,
                    "published": (r.published or r.updated).isoformat() if (r.published or r.updated) else None,
                    "authors": [a.name for a in r.authors],
                    "categories": list(r.categories),
                    "title": r.title,
                },
            }
        )

    return {**state, "docs": docs}


def node_chunk(state: PipelineState) -> PipelineState:
    docs = state.get("docs", [])
    chunk_size = int(state.get("chunk_size", 512))
    overlap = int(state.get("overlap", 50))

    chunks: List[Dict[str, Any]] = []
    for d in docs:
        parts = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        for j, p in enumerate(parts):
            chunks.append(
                {
                    "text": p,
                    "doc_id": d["doc_id"],
                    "metadata": d["metadata"],
                    "chunk_id": f"{d['doc_id']}::chunk{j:04d}",
                }
            )
    return {**state, "chunks": chunks}


def node_retrieve(state: PipelineState) -> PipelineState:
    # Import faiss inside the node to avoid segfaults on some macOS/Python combos
    # when importing at interpreter startup.
    import faiss  # type: ignore
    if not state.get("build_embeddings", True):
        return {**state, "hits": []}

    # Lazy imports to avoid heavy deps until needed
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore

    query = state["query"]
    chunks = state.get("chunks", [])
    if not chunks:
        return {**state, "hits": []}
    faiss_per_query = int(state.get("faiss_per_query", 80))
    final_docs = int(state.get("final_docs", 12))
    per_doc_chunks = int(state.get("per_doc_chunks", 1))
    recency_alpha = float(state.get("recency_alpha", 0.35))

    EMB_MODEL = CONFIG["rag"][3]["embedding_model"]
    emb = SentenceTransformer(EMB_MODEL, trust_remote_code=True)
    chunk_texts = [c["text"] for c in chunks]
    if len(chunk_texts) == 0:
        return {**state, "hits": []}
    X = emb.encode(chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    X = np.asarray(X, dtype="float32")
    if X.ndim != 2 or X.shape[0] == 0:
        return {**state, "hits": []}

    dim = int(X.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _retrieve_one(q: str, faiss_k: int = 80):
        qv = emb.encode([q], normalize_embeddings=True)
        D, I = index.search(np.asarray(qv, dtype="float32"), faiss_k)
        cands = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
        if not cands:
            return []
        pairs = [(q, c["text"]) for c in cands]
        scores = reranker.predict(pairs)
        return list(zip(cands, scores))

    def _aggregate_by_doc(paired):
        best = {}
        for ch, s in paired:
            did = ch["doc_id"]
            if (did not in best) or (s > best[did][1]):
                best[did] = (ch, s)
        return best

    all_pairs = []
    all_pairs.extend(_retrieve_one(query, faiss_k=faiss_per_query))

    best_by_doc = _aggregate_by_doc(all_pairs)
    if not best_by_doc:
        return {**state, "hits": []}

    raw_scores = np.array([s for (_, s) in best_by_doc.values()], dtype="float32")
    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())

    def _norm(x):
        return 0.0 if s_max == s_min else (x - s_min) / (s_max - s_min)

    scored_docs = []
    for did, (ch, rel_s) in best_by_doc.items():
        m = ch["metadata"]
        rec_s = _recency_score(m.get("published"))
        comb = (1 - recency_alpha) * _norm(rel_s) + recency_alpha * rec_s
        scored_docs.append((did, ch, rel_s, rec_s, comb))

    scored_docs.sort(key=lambda t: t[4], reverse=True)
    top_docs = scored_docs[:final_docs]

    hits: List[Dict[str, Any]] = []
    for did, best_chunk, rel_s, rec_s, comb in top_docs:
        b = dict(best_chunk)
        b["rel_score"] = float(rel_s)
        b["recency_score"] = float(rec_s)
        b["combined_score"] = float(comb)
        hits.append(b)
        if per_doc_chunks > 1:
            base_id = best_chunk["chunk_id"]
            try:
                base_idx = int(base_id.split("::chunk")[-1])
                same_doc = [c for c in chunks if c["doc_id"] == did]
                neighbors = [
                    c
                    for c in same_doc
                    if abs(int(c["chunk_id"].split("::chunk")[-1]) - base_idx) <= 2
                ]
                neighbors = [c for c in neighbors if c["chunk_id"] != base_id]
                for nb in neighbors[: max(0, per_doc_chunks - 1)]:
                    nb_annot = dict(nb)
                    nb_annot["rel_score"] = float(rel_s)
                    nb_annot["recency_score"] = float(rec_s)
                    nb_annot["combined_score"] = float(comb)
                    hits.append(nb_annot)
            except Exception:
                pass

    return {**state, "hits": hits}


def node_answer_and_save(state: PipelineState) -> PipelineState:
    hits = state.get("hits", [])
    query = state["query"]
    save_npz = state.get("save_npz")

    context = format_context(hits)
    final_answer = answer_query_with_context(query, context, model=LLM_MODEL)

    # Print and save artifacts (summary per doc, final answer, etc.)
    pretty_print_docs(hits, save_npz_path=save_npz, final_answer=final_answer, question=query)

    return {**state, "context": context, "final_answer": final_answer}


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("defaults", _init_defaults)
    g.add_node("keywords", node_keywords)
    g.add_node("search", node_arxiv_search_and_download)
    g.add_node("docs", node_build_docs)
    g.add_node("chunk", node_chunk)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer_and_save)

    g.add_edge(START, "defaults")
    g.add_edge("defaults", "keywords")
    g.add_edge("keywords", "search")
    g.add_edge("search", "docs")
    g.add_edge("docs", "chunk")
    g.add_edge("chunk", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

    return g.compile()


def run_pipeline(
    query: Optional[str] = None,
    max_days: Optional[int] = None,
    build_embeddings: Optional[bool] = None,
    faiss_per_query: Optional[int] = None,
    final_docs: Optional[int] = None,
    per_doc_chunks: Optional[int] = None,
    recency_alpha: Optional[float] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    save_npz: Optional[str] = None,
):
    app = build_graph()
    init: PipelineState = {}
    if query is not None:
        init["query"] = query
    if max_days is not None:
        init["max_days"] = max_days
    if build_embeddings is not None:
        init["build_embeddings"] = build_embeddings
    if faiss_per_query is not None:
        init["faiss_per_query"] = faiss_per_query
    if final_docs is not None:
        init["final_docs"] = final_docs
    if per_doc_chunks is not None:
        init["per_doc_chunks"] = per_doc_chunks
    if recency_alpha is not None:
        init["recency_alpha"] = recency_alpha
    if chunk_size is not None:
        init["chunk_size"] = chunk_size
    if overlap is not None:
        init["overlap"] = overlap
    if save_npz is not None:
        init["save_npz"] = save_npz

    final_state = app.invoke(init)
    return final_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LangGraph RAG pipeline")
    parser.add_argument("--query", type=str, default=None, help="Natural language query")
    parser.add_argument("--max-days", type=int, default=None, help="Search window in days")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable embeddings + retrieval")
    parser.add_argument("--faiss-per-query", type=int, default=None)
    parser.add_argument("--final-docs", type=int, default=None)
    parser.add_argument("--per-doc-chunks", type=int, default=None)
    parser.add_argument("--recency-alpha", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--save-npz", type=str, default=None)

    args = parser.parse_args()

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

    # Minimal console output summary
    print("[LangGraph] Completed. Summary:")
    print({k: v for k, v in state.items() if k in ("query", "keywords", "arxiv_query", "final_answer", "save_npz")})
