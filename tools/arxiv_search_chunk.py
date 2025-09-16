from datetime import datetime, time, timedelta
from urllib import response
from xml.parsers.expat import model
from dateutil import tz
import arxiv
import tiktoken
import re, requests
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
import faiss
from collections import defaultdict
from datetime import datetime
import time as time_module  # for retry sleeps; avoid clashing with datetime.time
from dateutil import tz
import numpy as np
import tiktoken
import os
import yaml, json
from pathlib import Path
# from dotenv import load_dotenv
# load_dotenv()
from content_parsing import extract_pdf_components, texts_to_plaintext

config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
# print(config['llm'][1]['model'])
provider = config['llm'][0]['provider']

if provider == 'openrouter':
    api_key = os.getenv("OPENROUTER_API_KEY")
elif provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
elif provider == "google":
    api_key = os.getenv("GOOGLE_API_KEY")
else:
    raise ValueError(f"Unsupported LLM provider: {provider}")

# def get_keywords_from_llm(natural_language_query: str, model: str = "google/gemini-2.0-flash-exp:free") -> list[str]:
def get_keywords_from_llm(natural_language_query: str, model: str = None) -> list[str]:
    """
    Uses an LLM (OpenRouter | OpenAI | Google Gemini) to extract keywords and phrases from a natural language query.

    Args:
        natural_language_query: The user's research question or topic.
+        model: The model name. For provider='openrouter', pass OpenRouter model id.
+               For provider='openai' pass OpenAI model id (e.g., 'gpt-4o-mini').
+               For provider='google' pass Gemini model id (e.g., 'gemini-1.5-flash' or 'gemini-2.0-flash-exp').

    Returns:
        A list of keywords and phrases suitable for an arXiv search.
    """
    # Ensure API key is set for the chosen provider
    env_hint = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider, "API_KEY")
    if not api_key:
        raise ValueError(f"No API key set for provider '{provider}'. Please export {env_hint}.")

    def _normalize_model_for_provider(p: str, m: str | None) -> str | None:
        if not m:
            return None
        m = m.strip()
        # If user accidentally passes OpenRouter-style names to direct providers, trim prefixes/suffixes
        if p in ("openai", "google"):
            if "/" in m:
                m = m.split("/")[-1]   # keep last segment
            if ":" in m:
                m = m.split(":")[0]    # drop qualifiers like ':free'
        return m

    model = _normalize_model_for_provider(provider, model)

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

    max_retries = 5
    backoff_base = 1.6

    def _sleep(attempt: int, retry_after: str | None):
        if retry_after and retry_after.isdigit():
            wait = float(retry_after)
        else:
            wait = backoff_base ** attempt
        time_module.sleep(min(wait, 30))

    def _parse_keywords(text: str) -> list[str]:
        return [kw.strip() for kw in (text or "").split(",") if kw.strip()]

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    # optional but recommended by OpenRouter
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                llm_output = data["choices"][0]["message"]["content"]

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                llm_output = data["choices"][0]["message"]["content"]

            elif provider == "google":
                # Google Generative Language API (Gemini)
                g_model = model or "gemini-1.5-flash"
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {"temperature": 0},
                }
                response = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                # Join all text parts from the top candidate
                parts = data["candidates"][0]["content"]["parts"]
                llm_output = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))

            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            keywords = _parse_keywords(llm_output)
            print(f"[INFO] LLM ({provider}:{model}) extracted keywords: {keywords}")
            return keywords

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            body = e.response.text[:500] if e.response is not None else ""
            print(f"[ERROR] HTTP {status} from {provider}: {body}")
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                _sleep(attempt, e.response.headers.get("Retry-After") if e.response else None)
                continue
            return []
        except (KeyError, IndexError, ValueError) as e:
            print(f"[ERROR] Failed to parse {provider} response: {e}")
            try:
                print(f"Raw response: {response.text[:500]}")
            except Exception:
                pass
            return []
        except requests.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            if attempt < max_retries:
                _sleep(attempt, None)
                continue
            return []

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

# while True:
#     natural_language_query = input("Please enter your research query (or press Enter to exit): ")
#     if not natural_language_query.strip():
#         # If the user just presses Enter, exit the script gracefully.
#         # Or, you could print a message and continue the loop.
#         print("No query provided. Exiting.")
#         exit()
#     else:
#         # If input is provided, break the loop and proceed.
#         break
try:
    natural_language_query = config['input'][0]['query']
except Exception as e:
    raise SystemExit("[ERROR] Could not read 'query' from config.yaml. Exiting.")

# Use the LLM to get keywords
llm_model = config['llm'][1]['model']
keywords = get_keywords_from_llm(natural_language_query, model=llm_model)

# Format the keywords into the final arXiv query string
QUERY = format_arxiv_query(keywords)

if not QUERY:
    raise SystemExit("[ERROR] Could not generate a valid query from the LLM. Exiting.")

print(f"[INFO] Generated arXiv Query: {QUERY}")

def download_arxiv_pdfs(pdf_url: str = None, output_dir: str = config['source'][0]['pdf_path']):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if pdf_url:
        pdf_response = requests.get(pdf_url)
        pdf_response.raise_for_status()
        pdf_filename = pdf_url.split("/")[-1]
        pdf_filename += '.pdf'
        pdf_path = Path(output_dir) / pdf_filename
        with open(pdf_path, 'wb') as f:
            f.write(pdf_response.content)
        print(f"Downloaded: {pdf_filename}")
    else:
        print("No PDF URL provided.")

    # response = requests.get(ARXIV_API_URL, params=params)
    # response.raise_for_status()

    # entries = response.json().get('feed', {}).get('entry', [])
    # for entry in entries:
    #     pdf_url = entry.get('link', [])[1].get('href')  # Get the PDF link
    #     title = entry.get('title', '').replace('/', '_').replace('\\', '_')  # Clean title for filename
    #     pdf_filename = f"{title}.pdf"
    #     pdf_path = Path(output_dir) / pdf_filename

    #     if not pdf_path.exists():
    #         pdf_response = requests.get(pdf_url)
    #         pdf_response.raise_for_status()
    #         with open(pdf_path, 'wb') as f:
    #             f.write(pdf_response.content)
    #         print(f"Downloaded: {pdf_filename}")
    #     else:
    #         print(f"File already exists: {pdf_filename}")

tz_london = tz.gettz("Europe/London")
now_local = datetime.now(tz_london)
days = config['input'][1]['max_days']
start_date = (now_local.date() - timedelta(days=days-1))  # 含今天共100天
start_dt = datetime.combine(start_date, time(0, 0, tzinfo=tz_london))
end_dt   = datetime.combine(now_local.date(), time(23, 59, 59, tzinfo=tz_london))

USE_UPDATED = False

client = arxiv.Client(page_size=100, delay_seconds=3)  # 分页+限速
search = arxiv.Search(
    query=QUERY,
    sort_by=arxiv.SortCriterion.SubmittedDate,  # ✅ 注意是 SortCriterion
    sort_order=arxiv.SortOrder.Descending
)

def in_window(dt_utc):
    dt_local = dt_utc.astimezone(tz_london)
    return start_dt <= dt_local <= end_dt

results = []
for r in client.results(search):
    ts = r.updated if USE_UPDATED else r.published
    if in_window(ts):
        results.append(r)
    else:
        # 提交时间按降序；一旦早于窗口起点就可以停止
        ts_local = ts.astimezone(tz_london)
        if ts_local < start_dt:
            break

print(f"Found {len(results)} results between {start_dt.date()} and {end_dt.date()} (Europe/London).")
for i, r in enumerate(results, 1):
    authors = ", ".join(a.name for a in r.authors)
    when_local = (r.updated if USE_UPDATED else r.published).astimezone(tz_london).strftime("%Y-%m-%d %H:%M")
    pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
    cats = ",".join(r.categories)
    abstract = " ".join(r.summary.split())  # 去掉多余换行/空白
    
    print(f"[{i}] {r.title}\n"
          f"    Authors: {authors}\n"
          f"    Time(UK): {when_local}\n"
          f"    Cats: {cats}\n"
          f"    PDF: {pdf_url}\n"
        #   f"    Abs: {r.entry_id}\n"
          f"    Abstract: {abstract}\n")
    download_arxiv_pdfs(pdf_url=pdf_url, output_dir=config['source'][0]['pdf_path'])

# not needed as we are using content_parsing.py

# def _strip_hyphenation(t: str) -> str:
#     return re.sub(r"-\s*\n\s*", "", t)

# def _normalize_ws(t: str) -> str:
#     t = t.replace("\r\n", "\n").replace("\r", "\n")
#     t = re.sub(r"\n{3,}", "\n\n", t)  # 多个空行压成一个
#     paras = [re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in t.split("\n\n")]
#     paras = [re.sub(r"\s{2,}", " ", p) for p in paras if p]
#     return "\n\n".join(paras).strip()

# def _cut_refs(t: str) -> str:
#     patt = re.compile(r"\n\s*(references|bibliography|acknowledg(e)?ments)\s*\n", re.I)
#     last = None
#     for m in patt.finditer("\n"+t+"\n"):
#         last = m
#     return t[:last.start()].strip() if last else t

# def extract_pdf_text(pdf_url: str) -> str:
#     try:
#         r = requests.get(pdf_url, timeout=60)
#         r.raise_for_status()
#         reader = PdfReader(BytesIO(r.content))
#         pages = [(p.extract_text() or "") for p in reader.pages]
#         raw = "\n\n".join(pages)
#         raw = _strip_hyphenation(raw)
#         raw = _normalize_ws(raw)
#         raw = _cut_refs(raw)
#         return raw
#     except Exception as e:
#         print(f"[WARN] PDF extract failed: {e}")
#         return ""

# --- Step 2: normalize docs from your `results` ---
docs = []
for r in results: 
    arxiv_id = r.entry_id.split("/")[-1]
    pdf_url  = r.pdf_url or r.entry_id.replace("abs", "pdf")
    title_abs = f"{r.title}\n\n{(' '.join(r.summary.split())).strip()}"
    # pdf_text = extract_pdf_text(pdf_url)
    # pdf_file = open(config['source'][0]['pdf_path'] + '/' + pdf_url, 'rb')
    pdf_filename = pdf_url.split("/")[-1]
    pdf_filename += '.pdf'
    pdf_path = Path(config['source'][0]['pdf_path']) / pdf_filename
    # pdf_text = extract_pdf_components(pdf_path)['texts']
    try:
        comps = extract_pdf_components(pdf_path)
        # Convert Docling items to a single cleaned plaintext (cuts after References, normalizes)
        body_text = texts_to_plaintext(comps['texts'])
    except Exception as e:
        print(f"[WARN] Docling parse failed for {pdf_filename}: {e}")
        body_text = ""
    # full_text = (title_abs + ("\n\n" + body_text if body_text else "")).strip()
    full_text = ("search_document: " + title_abs + ("\n\n" + body_text if body_text else "")).strip() ## to work with nomic-ai/nomic-embed-text-v1.5 embedding model
    # print(full_text)

    docs.append({
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
        }
    })

print(f"[INFO] Prepared {len(docs)} docs; with PDF text for {sum(1 for d in docs if len(d['text'])>len(d['title'])+20)} docs.")

# --- Step 3: chunking (token-based) ---
## TODO: save chunk => Ritesh
enc = tiktoken.get_encoding("cl100k_base")

chunk_size = config['rag'][4]['chunk_size']
overlap = config['rag'][5]['overlap']

def chunk_text(text: str, chunk_size: int = chunk_size, overlap: int = overlap):
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
    parts = chunk_text(d["text"], chunk_size, overlap)
    for j, p in enumerate(parts):
        chunks.append({
            "text": p,
            "doc_id": d["doc_id"],
            "metadata": d["metadata"],
            "chunk_id": f"{d['doc_id']}::chunk{j:04d}"
        })

print(f"[INFO] Total chunks: {len(chunks)}")

# --- Step 4: embeddings ---

# EMB_MODEL = "BAAI/bge-small-en-v1.5"
EMB_MODEL = config['rag'][3]['embedding_model']
emb = SentenceTransformer(EMB_MODEL, trust_remote_code=True)

chunk_texts = [c["text"] for c in chunks]
X = emb.encode(chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
X = np.asarray(X, dtype="float32")
print("[INFO] Embeddings:", X.shape)

# --- Step 5: FAISS index ---
dim = X.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X)
print("[INFO] Index size:", index.ntotal)

# --- Step 6: Retrieval + Rerank ---
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # 轻量高性价比

# --- Step 6b: Multi-query + doc-level rerank + recency boost ---

'''
faiss_per_query：召回越多越不漏，但重排更慢；80–200 较常见。

final_docs：最终要让 LLM看的论文数量。8–15 比较易读。

alpha_recency：偏新近性的程度；0.3–0.5 常用。

half_life_days：领域更新快就取小些（如 45–60 天），保守些就 90 天。

per_doc_chunks：若希望 LLM抓到更多细节，可设为 2，在每篇里再取相邻一个段落。
'''

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
def pretty_print_docs(hits, max_chars=300, save_npz_path=None):
    seen = set()
    for h in hits:
        did = h["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        m = h["metadata"]
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
        print(f"    Snip  : {snip}\n")

    """
    打印文档级摘要；若提供 save_npz_path，则将“arXiv ID -> 该文档对应的 out_hits（chunks）”
    保存为 .npz 文件，便于后续快速载入与复用。
    .npz 内容（按排名顺序对齐）：
      - 'arxiv_ids' : shape (N,)            # 每个文档的 arXiv ID
      - 'chunk_texts' : shape (N,) object  # 若 include_text=True，则保存每个chunk的文本
    备注：由于 'chunk_ids' / 'chunk_texts' 是变长列表，保存为 dtype=object 的数组；
         读取时需要 np.load(..., allow_pickle=True)。
    """
    # 先按照 hits 中出现的顺序将 chunk 聚合到“文档(arXiv ID)”维度
    doc_order = []           # 记录文档出现顺序（用于对齐保存）
    doc2chunks = {}          # {arxiv_id: [{'chunk_id':..., 'text':...}, ...]}
       
    for h in hits:
        aid = h["metadata"].get("arxiv_id") or h["doc_id"]  # 两者等价，这里偏向使用 arxiv_id
        if aid not in doc2chunks:
            doc2chunks[aid] = []
            doc_order.append(aid)
        doc2chunks[aid].append({
            "chunk_id": h["chunk_id"],
            "text": h["text"],
        })
    
    # 若指定保存路径，则写入 .npz
    if save_npz_path:
        arxiv_ids = np.array(doc_order, dtype=object)
        chunk_ids = np.array([[c["chunk_id"] for c in doc2chunks[aid]] for aid in doc_order], dtype=object)
        save_kwargs = {
            "arxiv_ids": arxiv_ids,
            "chunk_ids": chunk_ids,
        }
        chunk_texts = np.array([[c["text"] for c in doc2chunks[aid]] for aid in doc_order], dtype=object)
        save_kwargs["chunk_texts"] = chunk_texts
    
        np.savez(save_npz_path, **save_kwargs)
        print(f"[INFO] Saved mapping to {save_npz_path}. Load with: np.load('{save_npz_path}', allow_pickle=True)")
        
# === 用法示例 ===
# interest_queries = make_interest_queries(core="translational medicine", focus="edge computing")
hits = retrieve_recent_interest(
    natural_language_query,
    faiss_per_query=80,
    final_docs=12,
    per_doc_chunks=1,   # 每篇1段，便于LLM摘要
    alpha_recency=0.35  # 越大越偏新
)
pretty_print_docs(hits, save_npz_path="arxiv_out_hits.npz")

# def load_hits_from_npz(npz_path: str):
#     data = np.load(npz_path, allow_pickle=True)
#     arxiv_ids = data["arxiv_ids"]                # (N,)
#     chunk_ids_lists = data["chunk_ids"]          # (N,) object -> list[str]
#     chunk_texts_lists = data.get("chunk_texts")  # (N,) object -> list[str] or None

#     hits = []
#     for i, aid in enumerate(arxiv_ids):
#         aid = str(aid)
#         cids = list(chunk_ids_lists[i])
#         if chunk_texts_lists is None:
#             # 没有文本，只能先返回“占位”hits（见方式 B 回灌做法）
#             for cid in cids:
#                 hits.append({
#                     "text": None,                # 无文本
#                     "doc_id": aid,
#                     "chunk_id": cid,
#                 })
#         else:
#             texts = list(chunk_texts_lists[i])
#             # 与 cids 一一对应
#             for cid, t in zip(cids, texts):
#                 hits.append({
#                     "text": t,
#                     "doc_id": aid,
#                     "chunk_id": cid,
#                 })
#     return hits


# test = load_hits_from_npz("arxiv_out_hits.npz")

# --- Step 7: pack context ---

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