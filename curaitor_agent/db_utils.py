import os
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any, Optional


def init_db(db_path: str) -> None:
    path = Path(db_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path.as_posix()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
              arxiv_id TEXT PRIMARY KEY,
              title TEXT,
              published TEXT,
              pdf_url TEXT,
              authors TEXT,
              categories TEXT,
              entry_id TEXT,
              text TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hits (
              chunk_id TEXT PRIMARY KEY,
              doc_id TEXT,
              rel_score REAL,
              recency_score REAL,
              combined_score REAL,
              text TEXT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              query TEXT,
              final_answer TEXT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def upsert_docs(db_path: str, docs: Iterable[Dict[str, Any]]) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        n = 0
        for d in docs:
            meta = d.get("metadata", {})
            cur.execute(
                """
                INSERT OR REPLACE INTO papers (arxiv_id, title, published, pdf_url, authors, categories, entry_id, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta.get("arxiv_id") or d.get("doc_id"),
                    meta.get("title") or d.get("title"),
                    meta.get("published"),
                    meta.get("pdf_url"),
                    json.dumps(meta.get("authors", []), ensure_ascii=False),
                    json.dumps(meta.get("categories", []), ensure_ascii=False),
                    meta.get("entry_id"),
                    d.get("text"),
                ),
            )
            n += 1
        conn.commit()
        return n


def upsert_hits(db_path: str, hits: Iterable[Dict[str, Any]]) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        n = 0
        for h in hits:
            cur.execute(
                """
                INSERT OR REPLACE INTO hits (chunk_id, doc_id, rel_score, recency_score, combined_score, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    h.get("chunk_id"),
                    h.get("doc_id"),
                    float(h.get("rel_score", 0.0)) if h.get("rel_score") is not None else None,
                    float(h.get("recency_score", 0.0)) if h.get("recency_score") is not None else None,
                    float(h.get("combined_score", 0.0)) if h.get("combined_score") is not None else None,
                    h.get("text"),
                ),
            )
            n += 1
        conn.commit()
        return n


def insert_answer(db_path: str, query: str, final_answer: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO answers (query, final_answer) VALUES (?, ?)",
            (query, final_answer),
        )
        conn.commit()
        return cur.lastrowid

