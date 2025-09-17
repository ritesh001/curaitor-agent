This is a sample MCP server that provides tools to search for papers on arXiv
# and extract information about specific papers. It uses the `arxiv` library to
# Nanta, Shichuan
# Sep 2025
import arxiv
import asyncio
import importlib.util
import sys
notif_spec = importlib.util.spec_from_file_location("Notifications", "./Notifications.Py")
Notifications = importlib.util.module_from_spec(notif_spec)
sys.modules["Notifications"] = Notifications
notif_spec.loader.exec_module(Notifications)
NotificationRequest = Notifications.NotificationRequest
NotificationDispatcher = Notifications.NotificationDispatcher
import json
import os
import requests
import csv
from typing import List
from mcp.server.fastmcp import FastMCP
from datetime import datetime, time, timedelta
from dateutil import tz


from arxiv_search_chunk import get_keyworks_from_llm


# from pathlib import Path
# import argparse

PAPER_DIR = "data/agent_papers"

mcp = FastMCP("paper_search")
# ...existing code...

@mcp.tool()
def send_notification(notification_types: list[str] | str, message: str, email: str = None, phone: str = None, slack_user: str = None, wechat_user: str = None) -> dict:
    """
    Send a notification to the user via the selected channel(s).
    Args:
        notification_types: List or string of channels ('email', 'whatsapp', 'wechat', 'slack', or 'auto').
        message: The message to send.
        email: Email address for email notifications.
        phone: Phone number for WhatsApp notifications.
        slack_user: Slack user ID for Slack notifications.
        wechat_user: WeChat user ID for WeChat notifications.
    Returns:
        Dictionary with status for each channel.
    """
    req = NotificationRequest(
        notification_types=notification_types,
        message=message,
        email=email,
        phone=phone,
        slack_user=slack_user,
        wechat_user=wechat_user
    )
    # Run the async dispatcher in sync context
    return asyncio.run(NotificationDispatcher.dispatch(req))

@mcp.tool()
def get_keywords_from_llm(natural_language_query: str, model: str = "google/gemini-2.0-flash-exp:free") -> list[str]:
    """
    Uses an LLM via OpenRouter to extract relevant keywords and phrases from a natural language query.

    Args:
        natural_language_query: The user's research question or topic.
        model: The OpenRouter model to use (defaults to a free one).

    Returns:
        A list of keywords and phrases suitable for an arXiv search.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    # A detailed prompt telling the LLM exactly what to do.
    prompt = f"""
    You are an expert academic researcher specializing in chemistry, biology, physics, computer science, and condensed matter physics.
    Your task is to analyze the following user query and extract a list of precise, effective keywords and multi-word phrases for searching the arXiv database.
    The goal is to find the most relevant academic papers.

    - Identify core concepts, technical terms, and important named entities.
    - For multi-word concepts (e.g., "machine learning potential"), keep them as a single phrase.
    - Do not include generic words.
    - Return ONLY a comma-separated list of these keywords and phrases. Do not add any explanation or introductory text.

    User Query: "{natural_language_query}"

    Keywords:
    """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Extract the text content from the response
        llm_output = response.json()['choices'][0]['message']['content']

        # Clean up the output and split it into a list of keywords
        keywords = [kw.strip() for kw in llm_output.split(',') if kw.strip()]
        
        print(f"[INFO] LLM extracted keywords: {keywords}")
        return keywords

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return []
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Failed to parse LLM response: {e}")
        print(f"Raw response: {response.text}")
        return []

@mcp.tool()
def get_one_sentence_summary(text: str, model: str = "google/gemini-flash-1.5") -> str:
    """Uses an LLM to generate a one-sentence summary of a paper's main contribution."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "[ERROR] OPENROUTER_API_KEY not set."

    prompt = f"""
    You are an expert academic editor. Your task is to read the following text from a research paper and synthesize its core contribution into a single, concise sentence.

    The summary MUST follow this structure, combining these four elements:
    1.  **Problem:** What question or problem is the paper trying to solve?
    2.  **Method:** What is the primary method, model, or approach used?
    3.  **System:** What specific system, material, or example is investigated?
    4.  **Outcome:** What is the main finding or result?

    Combine these points into one fluid sentence. For example: "To address [Problem], this paper introduces [Method] to study [System], demonstrating that [Outcome]."

    Do not add any preamble or explanation. Return ONLY the single summary sentence.

    Text:
    \"\"\"
    {text}
    \"\"\"

    One-sentence summary:
    """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]})
        )
        response.raise_for_status()
        summary = response.json()['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        print(f"[WARN] Summarization failed: {e}")
        return "Could not generate summary."

@mcp.tool()
def format_arxiv_query(keywords: list[str], field: str = "all", max_keywords: int = 3) -> str:
    """
    Formats a list of keywords into a valid arXiv search query string.
    Uses OR to connect keywords and quotes for multi-word phrases.
    Limits the number of keywords to avoid hitting URL length limits.
    """
    if not keywords:
        return ""
    
    # Take only the top N most relevant keywords (LLM usually returns them in order)
    limited_keywords = keywords[:max_keywords]
    
    formatted_keywords = []
    for kw in limited_keywords:
        if ' ' in kw:
            formatted_keywords.append(f'"{kw}"')  # Add quotes for phrases
        else:
            formatted_keywords.append(kw)
    
    # Use OR to cast a wider net. The RAG pipeline will handle the filtering.
    return f"{field}:(" + " OR ".join(formatted_keywords) + ")"


# --- Step 1: Define a Natural Language Query and Generate Keywords with LLM ---

# natural_language_query = "I want to find papers about using machine learning potentials to accelerate molecular dynamics simulations for material science."

while True:
    natural_language_query = input("Please enter your research query (or press Enter to exit): ")
    if not natural_language_query.strip():
        # If the user just presses Enter, exit the script gracefully.
        # Or, you could print a message and continue the loop.
        print("No query provided. Exiting.")
        exit()
    else:
        # If input is provided, break the loop and proceed.
        break


# Use the LLM to get keywords
# You can choose a different model from OpenRouter if you wish, e.g., "google/gemini-flash-1.5"
keywords = get_keywords_from_llm(natural_language_query, model="deepseek/deepseek-r1:free")

# Format the keywords into the final arXiv query string
QUERY = format_arxiv_query(keywords)

if not QUERY:
    raise SystemExit("[ERROR] Could not generate a valid query from the LLM. Exiting.")

print(f"[INFO] Generated arXiv Query: {QUERY}")


# The `make_interest_queries` function is now used for the RAG retrieval step later on.
# We can make it use our LLM-generated keywords.
def make_interest_queries(keywords_list):
    # The function now simply returns the list of keywords generated by the LLM.
    # You could add more complex logic here if needed.
    return keywords_list



# --- Step 2: Use the arxiv library to search and filter results by date ---
tz_london = tz.gettz("Europe/London") # arXiv uses UTC but we want to filter by London time
now_local = datetime.now(tz_london) # current time in London
days = 100 # look back this many days
start_date = (now_local.date() - timedelta(days=days-1))  # inclusive
start_dt = datetime.combine(start_date, time(0, 0, tzinfo=tz_london)) # start of day
end_dt   = datetime.combine(now_local.date(), time(23, 59, 59, tzinfo=tz_london)) # end of day

# Whether to use the 'updated' timestamp instead of 'published' for filtering.
# 'updated' reflects the latest version, which may be more relevant for recent changes.
USE_UPDATED = False

client = arxiv.Client(page_size=100, delay_seconds=7)  # might need to increase delay_seconds if you get HTTP 429
search = arxiv.Search(
    query=QUERY,
    sort_by=arxiv.SortCriterion.SubmittedDate,  # it is SortCriterion
    sort_order=arxiv.SortOrder.Descending
)

@mcp.tool()
def in_window(dt_utc):
    dt_local = dt_utc.astimezone(tz_london)
    return start_dt <= dt_local <= end_dt

results = []
# --- MODIFICATION START ---
# Wrap the result fetching in a try...except block to handle the library's pagination bug.
try:
    for r in client.results(search):
        ts = r.updated if USE_UPDATED else r.published
        if in_window(ts):
            results.append(r)
        else:
            # 提交时间按降序；一旦早于窗口起点就可以停止
            ts_local = ts.astimezone(tz_london)
            if ts_local < start_dt:
                print("[INFO] Reached end of date window. Stopping search.")
                break
except arxiv.UnexpectedEmptyPageError:
    # This error occurs when the total number of results is an exact multiple of the page_size.
    # It's safe to ignore it and proceed with the results we have.
    print("[INFO] Caught UnexpectedEmptyPageError. This is expected if total results are a multiple of page size. Continuing.")
# --- MODIFICATION END ---


print(f"Found {len(results)} results between {start_dt.date()} and {end_dt.date()} (Europe/London).")
for i, r in enumerate(results, 1):
    authors = ", ".join(a.name for a in r.authors)
    when_local = (r.updated if USE_UPDATED else r.published).astimezone(tz_london).strftime("%Y-%m-%d %H:%M")
    pdf = r.pdf_url or r.entry_id.replace("abs", "pdf")
    cats = ",".join(r.categories)
    abstract = " ".join(r.summary.split())  # 去掉多余换行/空白
    
    print(f"[{i}] {r.title}\n"
          f"    Authors: {authors}\n"
          f"    Time(UK): {when_local}\n"
          f"    Cats: {cats}\n"
          f"    PDF: {pdf}\n"
          f"    Abs: {r.entry_id}\n"
          f"    Abstract: {abstract}\n")



# PDF 抽取与清洗（pypdf）
# --- Step 1: PDF download + text extract + clean ---
import re, requests
from io import BytesIO
from pypdf import PdfReader

@mcp.tool()
def _strip_hyphenation(t: str) -> str:
    return re.sub(r"-\s*\n\s*", "", t)

@mcp.tool()
def _normalize_ws(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)  # 多个空行压成一个
    paras = [re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in t.split("\n\n")]
    paras = [re.sub(r"\s{2,}", " ", p) for p in paras if p]
    return "\n\n".join(paras).strip()

@mcp.tool()
def _cut_refs(t: str) -> str:
    patt = re.compile(r"\n\s*(references|bibliography|acknowledg(e)?ments)\s*\n", re.I)
    last = None
    for m in patt.finditer("\n"+t+"\n"):
        last = m
    return t[:last.start()].strip() if last else t

@mcp.tool()
def extract_pdf_text(pdf_url: str) -> str:
    try:
        r = requests.get(pdf_url, timeout=60)
        r.raise_for_status()
        reader = PdfReader(BytesIO(r.content))
        pages = [(p.extract_text() or "") for p in reader.pages]
        raw = "\n\n".join(pages)
        raw = _strip_hyphenation(raw)
        raw = _normalize_ws(raw)
        raw = _cut_refs(raw)
        return raw
    except Exception as e:
        print(f"[WARN] PDF extract failed: {e}")
        return ""
    
    

#② 规整文档（title+abstract，并尝试拼上全文）
# --- Step 2: normalize docs from your `results` ---
docs = []
for r in results: 
    arxiv_id = r.entry_id.split("/")[-1]
    pdf_url  = r.pdf_url or r.entry_id.replace("abs", "pdf")
    title_abs = f"{r.title}\n\n{(' '.join(r.summary.split())).strip()}"
    pdf_text = extract_pdf_text(pdf_url)
    full_text = (title_abs + ("\n\n" + pdf_text if pdf_text else "")).strip()

    docs.append({
        "doc_id": arxiv_id,
        "title": r.title,
        "text": full_text,
        "metadata": {
            "title": r.title, # Add this line
            "arxiv_id": arxiv_id,
            "entry_id": r.entry_id,
            "pdf_url": pdf_url,
            "published": (r.published or r.updated).isoformat() if (r.published or r.updated) else None,
            "authors": [a.name for a in r.authors],
            "categories": list(r.categories),
        }
    })

print(f"[INFO] Prepared {len(docs)} docs; with PDF text for {sum(1 for d in docs if len(d['text'])>len(d['title'])+20)} docs.")




#③ 切片（token级，500/100）
# --- Step 3: chunking (token-based) ---
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

@mcp.tool()
def chunk_text(text: str, chunk_size=500, overlap=100):
    if not text.strip():
        return []
    toks = enc.encode(text)
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(toks), step):
        window = toks[i:i+chunk_size]
        if not window: break
        chunks.append(enc.decode(window))
        if i + chunk_size >= len(toks):
            break
    return chunks or [text]

chunks = []
for d in docs:
    parts = chunk_text(d["text"], 500, 100)
    for j, p in enumerate(parts):
        chunks.append({
            "text": p,
            "doc_id": d["doc_id"],
            "metadata": d["metadata"],
            "chunk_id": f"{d['doc_id']}::chunk{j:04d}"
        })

print(f"[INFO] Total chunks: {len(chunks)}")


#④ 向量化（Sentence-Transformers，bge-small-en）
# --- Step 4: embeddings ---
from sentence_transformers import SentenceTransformer
import numpy as np

EMB_MODEL = "BAAI/bge-small-en-v1.5"  # 多语可换 "BAAI/bge-m3"
emb = SentenceTransformer(EMB_MODEL)

chunk_texts = [c["text"] for c in chunks]
X = emb.encode(chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
X = np.asarray(X, dtype="float32")
print("[INFO] Embeddings:", X.shape)



#⑤ 建索引（FAISS 内积；等价于余弦，因为已归一化）
# --- Step 5: FAISS index ---
import faiss
dim = X.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X)
print("[INFO] Index size:", index.ntotal)




#⑥ 检索 + 交叉编码器复排（MiniLM）
# --- Step 6: Retrieval + Rerank ---
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # 轻量高性价比

# --- Step 6b: Multi-query + doc-level rerank + recency boost ---

'''
faiss_per_query：召回越多越不漏，但重排更慢；80–200 较常见。

final_docs：最终要让 LLM看的论文数量。8–15 比较易读。

alpha_recency：偏新近性的程度；0.3–0.5 常用。

half_life_days：领域更新快就取小些（如 45–60 天），保守些就 90 天。

per_doc_chunks：若希望 LLM抓到更多细节，可设为 2，在每篇里再取相邻一个段落。
'''
from collections import defaultdict
from datetime import datetime
from dateutil import tz
import numpy as np



# Step 6) 单次查询的粗排召回（FAISS）+ 交叉编码器重排
def _retrieve_one(query, faiss_k=80):
    qv = emb.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(qv, dtype="float32"), faiss_k)
    cands = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    if not cands:
        return []
    pairs = [(query, c["text"]) for c in cands]
    scores = reranker.predict(pairs)  # 越大越相关
    return list(zip(cands, scores))

# Step 6) 以“文档”为单位聚合：每个 doc 取其最高相关的 chunk 得分
def _aggregate_by_doc(paired_list):
    best_by_doc = {}  # doc_id -> (chunk, score)
    for ch, s in paired_list:
        did = ch["doc_id"]
        if (did not in best_by_doc) or (s > best_by_doc[did][1]):
            best_by_doc[did] = (ch, s)
    return best_by_doc  # dict

# Step 6) 计算新近性分数（半衰期：60天，可调）
def _recency_score(iso_datetime: str, half_life_days=60, tz_str="Europe/London"):
    if not iso_datetime:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso_datetime)
    except Exception:
        return 0.0
    now = datetime.now(tz.gettz(tz_str))
    # 若 dt 无时区，按本地时区处理
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.gettz(tz_str))
    days = (now - dt).total_seconds() / 86400.0
    if days < 0:
        days = 0
    return 0.5 ** (days / half_life_days)  # 今天=1，60天≈0.5，120天≈0.25

# Step 6) 融合多查询：RRF + 新近性加权（alpha 可调）
def retrieve_recent_interest(
    queries,
    faiss_per_query=80,
    final_docs=12,
    per_doc_chunks=1,
    alpha_recency=0.35,  # 0~1，越大越偏向新文章
):
    # 多查询召回+重排
    all_pairs = []
    for q in queries:
        all_pairs.extend(_retrieve_one(q, faiss_k=faiss_per_query))

    if not all_pairs:
        return []

    # 文档粒度取最佳相关 score
    best_by_doc = _aggregate_by_doc(all_pairs)

    # 归一化相关度
    raw_scores = np.array([s for (_, s) in best_by_doc.values()], dtype="float32")
    if raw_scores.size == 0:
        return []
    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
    def _norm(x):
        return 0.0 if s_max == s_min else (x - s_min) / (s_max - s_min)

    # 计算新近性 & 融合分数
    scored_docs = []
    for did, (ch, rel_s) in best_by_doc.items():
        m = ch["metadata"]
        rec_s = _recency_score(m.get("published"))
        comb = (1 - alpha_recency) * _norm(rel_s) + alpha_recency * rec_s
        scored_docs.append((did, ch, rel_s, rec_s, comb))

    # 依综合分排序，保留 top 文档
    scored_docs.sort(key=lambda t: t[4], reverse=True)
    top_docs = scored_docs[:final_docs]

    # 为每个文档再挑选代表性 chunk（默认就取刚才的最佳chunk；也可以扩展取同文档下相邻chunk）
    out_hits = []
    for did, best_chunk, rel_s, rec_s, comb in top_docs:
        out_hits.append(best_chunk)
        # 如需每文档多片段，可在此处基于 chunk_id 邻近再取1-2段
        if per_doc_chunks > 1:
            # 示例：同文档内找到与 best_chunk 相邻的 chunk
            base_id = best_chunk["chunk_id"]
            try:
                base_idx = int(base_id.split("::chunk")[-1])
                same_doc = [c for c in chunks if c["doc_id"] == did]
                # 简单找相邻的几段
                neighbors = [c for c in same_doc if abs(int(c["chunk_id"].split("::chunk")[-1]) - base_idx) <= 2]
                # 去掉重复
                neighbors = [c for c in neighbors if c["chunk_id"] != base_id]
                # 最多补足 per_doc_chunks-1 个
                out_hits.extend(neighbors[:max(0, per_doc_chunks - 1)])
            except Exception:
                pass

    return out_hits

# Step 6) 友好打印（文档级）
def pretty_print_docs(hits, summaries, max_chars=500):
    seen = set()
    for h in hits:
        did = h["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        m = h["metadata"]
        summary = summaries.get(did, "Summary not available.") # Get the summary
        title = m.get("title", "") if "title" in m else ""
        authors = ", ".join(m.get("authors", [])) or "Unknown"
        when = m.get("published", "") or ""
        snip = h["text"].strip().replace("\n", " ")
        if len(snip) > max_chars: snip = snip[:max_chars] + " ..."
        print(f"[{len(seen)}] arXiv:{m.get('arxiv_id','')}")
        print(f"    Title : {title}")
        print(f"    Authors: {authors}")
        print(f"    Date  : {when}")
        print(f"    PDF   : {m.get('pdf_url','')}")
        print(f"    Summary: {summary}") # Print the summary
        print(f"    Snip  : {snip}\n")

def save_hits_to_csv(hits, summaries, filename="search_results.csv"):
    """Saves the top unique documents from the hits to a CSV file."""
    headers = ['arXiv_ID', 'Title', 'Authors', 'Published_Date', 'PDF_URL', 'One_Sentence_Summary', 'Snippet']
    seen = set()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for h in hits:
            did = h["doc_id"]
            if did in seen:
                continue
            seen.add(did)
            
            m = h["metadata"]
            summary = summaries.get(did, "Summary not available.") # Get the summary
            
            arxiv_id = m.get('arxiv_id', '')
            title = m.get("title", "")
            authors = ", ".join(m.get("authors", [])) or "Unknown"
            when = m.get("published", "") or ""
            pdf_url = m.get('pdf_url', '')
            snip = h["text"].strip().replace("\n", " ")

            writer.writerow([arxiv_id, title, authors, when, pdf_url, summary, snip]) # Add summary to the row
    
    print(f"\n[INFO] Successfully saved {len(seen)} results to {filename}")


# === 用法示例 ===

# --- Configuration for Summarization ---
# Set to True to use the full text of the paper for summarization (slower, more comprehensive).
# Set to False to use only the most relevant text chunk (faster, more focused).
SUMMARIZE_FULL_TEXT = False # need to be set in .yaml file

# interest_queries = make_interest_queries(core="translational medicine", focus="edge computing")
interest_queries = make_interest_queries(keywords)
hits = retrieve_recent_interest(
    interest_queries,
    faiss_per_query=80,
    final_docs=12,
    per_doc_chunks=1,   # 每篇1段，便于LLM摘要
    alpha_recency=0.35  # 越大越偏新
)

# --- NEW: Generate one-sentence summaries for the top hits ---

# Create a dictionary for quick lookup of full document text by doc_id
docs_by_id = {doc['doc_id']: doc for doc in docs}

doc_summaries = {}
unique_docs_for_summary = []
seen_docs = set()
for h in hits:
    if h['doc_id'] not in seen_docs:
        unique_docs_for_summary.append(h)
        seen_docs.add(h['doc_id'])

summary_mode = "full text" if SUMMARIZE_FULL_TEXT else "best chunk"
print(f"\n[INFO] Generating one-sentence summaries for {len(unique_docs_for_summary)} papers (using {summary_mode})...")

for i, h in enumerate(unique_docs_for_summary, 1):
    print(f"  > Summarizing paper {i}/{len(unique_docs_for_summary)}: {h['metadata']['title'][:50]}...")
    
    text_to_summarize = ""
    if SUMMARIZE_FULL_TEXT:
        # Use the full text from the original document
        full_doc = docs_by_id.get(h['doc_id'])
        if full_doc:
            text_to_summarize = full_doc['text']
    else:
        # Use the text of the most relevant chunk (original behavior)
        text_to_summarize = h['text']

    if text_to_summarize:
        summary = get_one_sentence_summary(text_to_summarize)
        doc_summaries[h['doc_id']] = summary
    else:
        doc_summaries[h['doc_id']] = "Text for summarization not found."

print("[INFO] Summarization complete.")
# --- END NEW SECTION ---

pretty_print_docs(hits, doc_summaries)
save_hits_to_csv(hits, doc_summaries)


#⑦ 组包
# --- Step 7: pack context ---
import tiktoken

def format_context(hits, max_ctx_tokens=1800, model_enc="cl100k_base"):
    enc = tiktoken.get_encoding(model_enc)
    parts, used = [], 0
    for h in hits:
        src = f"[Source: arXiv:{h['metadata']['arxiv_id']}]"
        block = f"{src}\n{h['text'].strip()}\n"
        tok = enc.encode(block)
        if used + len(tok) > max_ctx_tokens:
            break
        parts.append(block)
        used += len(tok)
    return "\n---\n".join(parts)

context = format_context(hits)



if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)
