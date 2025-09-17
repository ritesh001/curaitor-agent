from datetime import datetime, time, timedelta
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
import time as time_module  # for retry sleeps; avoid clashing with datetime.time
from dateutil import tz
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
raw_model = config['llm'][1]['model']

if provider == 'openrouter':
    api_key = os.getenv("OPENROUTER_API_KEY")
elif provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
elif provider == "google":
    api_key = os.getenv("GOOGLE_API_KEY")
else:
    raise ValueError(f"Unsupported LLM provider: {provider}")
if not api_key:
    raise ValueError(f"Missing API key for provider '{provider}'. Set the appropriate env var.")

def _normalize_model_for_provider(p: str, m: str | None) -> str | None:
    if not m:
        return None
    m = m.strip()
    # For direct providers, drop OpenRouter-style prefixes/suffixes
    if p in ("openai", "google"):
        if "/" in m:
            m = m.split("/")[-1]  # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
        if ":" in m:
            m = m.split(":")[0]   # e.g., "...:free" -> base model
    return m

LLM_MODEL = _normalize_model_for_provider(provider, raw_model)
# Reasonable fallbacks if model omitted+if not LLM_MODEL:
if not LLM_MODEL:
    LLM_MODEL = {
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-flash",
        "openrouter": "google/gemini-2.0-flash-exp:free",
    }.get(provider, None)
print(f"[INFO] Provider: {provider} | Model: {LLM_MODEL}")

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
    # natural_language_query = config['input'][0]['query']
    natural_language_query = 'search_query: ' + config['input'][0]['query'] ## to work with nomic-ai/nomic-embed-text-v1.5 embedding model
except Exception as e:
    raise SystemExit("[ERROR] Could not read 'query' from config.yaml. Exiting.")

# Use the LLM to get keywords
# llm_model = config['llm'][1]['model']
keywords = get_keywords_from_llm(natural_language_query, model=LLM_MODEL)

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
        ts_local = ts.astimezone(tz_london)
        if ts_local < start_dt:
            break

print(f"Found {len(results)} results between {start_dt.date()} and {end_dt.date()} (Europe/London).")
for i, r in enumerate(results, 1):
    authors = ", ".join(a.name for a in r.authors)
    when_local = (r.updated if USE_UPDATED else r.published).astimezone(tz_london).strftime("%Y-%m-%d %H:%M")
    pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
    cats = ",".join(r.categories)
    abstract = " ".join(r.summary.split()) 
    
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
            "title": r.title,
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
faiss_per_query

final_docs

alpha_recency

half_life_days

per_doc_chunks
'''

def _retrieve_one(query, faiss_k=80):
    qv = emb.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(qv, dtype="float32"), faiss_k)
    cands = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
    if not cands:
        return []
    pairs = [(query, c["text"]) for c in cands]
    scores = reranker.predict(pairs)  # 越大越相关
    return list(zip(cands, scores))

def _aggregate_by_doc(paired_list):
    best_by_doc = {}  # doc_id -> (chunk, score)
    for ch, s in paired_list:
        did = ch["doc_id"]
        if (did not in best_by_doc) or (s > best_by_doc[did][1]):
            best_by_doc[did] = (ch, s)
    return best_by_doc  # dict

def _recency_score(iso_datetime: str, half_life_days=60, tz_str="Europe/London"):
    if not iso_datetime:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso_datetime)
    except Exception:
        return 0.0
    now = datetime.now(tz.gettz(tz_str))
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.gettz(tz_str))
    days = (now - dt).total_seconds() / 86400.0
    if days < 0:
        days = 0
    return 0.5 ** (days / half_life_days)  # 今天=1，60天≈0.5，120天≈0.25

### processing by LLM

def answer_query_with_context(question: str, context: str, model: str | None = None) -> str:
    """
    Ask the selected LLM to answer using the retrieved arXiv context.
    Reuses the same `provider` + `api_key` as above.
    """
    sys_instructions = (
        "You are a concise research assistant. Answer using ONLY the provided arXiv context. "
        "Cite arXiv IDs in brackets like [arXiv:XXXX.XXXXX]. If the context is insufficient, say so briefly."
    )
    user_msg = f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer in 5-8 sentences."

    max_retries, backoff_base = 5, 1.6
    def _sleep(attempt: int, retry_after: str | None):
        wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff_base ** attempt
        time_module.sleep(min(wait, 30))

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After")); continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After")); continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "google":
                # Gemini: put instructions + question + context into one prompt
                g_model = model or "gemini-1.5-flash"
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                prompt = f"{sys_instructions}\n\n{user_msg}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                }
                resp = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After")); continue
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

            else:
                return "[ERROR] Unsupported provider."
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                _sleep(attempt, e.response.headers.get("Retry-After")); continue
            return f"[ERROR] QA request failed: {getattr(e.response,'text','')[:300]}"
        except Exception as e:
            if attempt < max_retries:
                _sleep(attempt, None); continue
            return f"[ERROR] QA request failed: {e}"

# Generate a direct answer to the initial query using retrieved context
# qa_model = LLM_MODEL 
# answer = answer_query_with_context(natural_language_query, context, model=qa_model)
# print("\n[ANSWER]\n" + answer + "\n")

def summarize_docs(docs, model: str | None = None) -> str:
    """
    Summarize a list of documents (chunks) into a concise overview using the selected LLM.
    """
    sys_instructions = (
        "You are an expert academic researcher. Summarize the key points from the provided arXiv document excerpts. "
        "Write a concise summary in 5-8 sentences."
    )
    combined_texts = "\n\n---\n\n".join(
        f"[arXiv:{d['metadata'].get('arxiv_id','')}]\n{d['text']}" for d in docs
    )
    user_msg = f"Document Excerpts:\n{combined_texts}\n\nSummary:"

    max_retries, backoff_base = 5, 1.6
    def _sleep(attempt: int, retry_after: str | None):
        wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff_base ** attempt
        time_module.sleep(min(wait, 30))

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After")); continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After")); continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif provider == "google":
                # Gemini: put instructions + question + context into one prompt
                g_model = model or "gemini-1.5-flash"
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                prompt = f"{sys_instructions}\n\n{user_msg}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                }
                resp = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After")); continue
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            else:
                return "[ERROR] Unsupported provider."
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                _sleep(attempt, e.response.headers.get("Retry-After")); continue
            return f"[ERROR] Summarization request failed: {getattr(e.response,'text','')[:300]}"
### finish

def retrieve_recent_interest(
    queries,
    faiss_per_query=80,
    final_docs=12,
    per_doc_chunks=1,
    alpha_recency=0.35,  # 0~1，越大越偏向新文章
):
    # Allow a single query string or a list of queries
    if isinstance(queries, str):
        queries = [queries]

    all_pairs = []
    for q in queries:
        all_pairs.extend(_retrieve_one(q, faiss_k=faiss_per_query))

    if not all_pairs:
        return []

    best_by_doc = _aggregate_by_doc(all_pairs)

    raw_scores = np.array([s for (_, s) in best_by_doc.values()], dtype="float32")
    if raw_scores.size == 0:
        return []
    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
    def _norm(x):
        return 0.0 if s_max == s_min else (x - s_min) / (s_max - s_min)

    scored_docs = []
    for did, (ch, rel_s) in best_by_doc.items():
        m = ch["metadata"]
        rec_s = _recency_score(m.get("published"))
        comb = (1 - alpha_recency) * _norm(rel_s) + alpha_recency * rec_s
        scored_docs.append((did, ch, rel_s, rec_s, comb))

    scored_docs.sort(key=lambda t: t[4], reverse=True)
    top_docs = scored_docs[:final_docs]

    out_hits = []
    for did, best_chunk, rel_s, rec_s, comb in top_docs:
        # out_hits.append(best_chunk)
        # annotate best chunk with doc-level scores
        best_annot = dict(best_chunk)
        best_annot["rel_score"] = float(rel_s)
        best_annot["recency_score"] = float(rec_s)
        best_annot["combined_score"] = float(comb)
        out_hits.append(best_annot)

        if per_doc_chunks > 1:

            base_id = best_chunk["chunk_id"]
            try:
                base_idx = int(base_id.split("::chunk")[-1])
                same_doc = [c for c in chunks if c["doc_id"] == did]
                neighbors = [c for c in same_doc if abs(int(c["chunk_id"].split("::chunk")[-1]) - base_idx) <= 2]
                neighbors = [c for c in neighbors if c["chunk_id"] != base_id]
                # out_hits.extend(neighbors[:max(0, per_doc_chunks - 1)])
                # carry scores on neighbors too (optional, copy best's scores)
                for nb in neighbors[:max(0, per_doc_chunks - 1)]:
                    nb_annot = dict(nb)
                    nb_annot["rel_score"] = float(rel_s)
                    nb_annot["recency_score"] = float(rec_s)
                    nb_annot["combined_score"] = float(comb)
                    out_hits.append(nb_annot)
            except Exception:
                pass

    return out_hits

# --- Step 7: pack context and get a direct answer ---
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

# def pretty_print_docs(hits, max_chars=300, save_npz_path=None):
    # seen = set()
    # for h in hits:
    #     did = h["doc_id"]
    #     if did in seen:
    #         continue
    #     seen.add(did)
    #     m = h["metadata"]
    #     title = m.get("title", "") if "title" in m else ""
    #     authors = ", ".join(m.get("authors", [])) or "Unknown"
    #     when = m.get("published", "") or ""
    #     print(h)
    #     # snip = h["text"].strip().replace("\n", " ")
    #     summary = summarize_docs([h], model=llm_model)
    #     snip = summary.replace("\n", " ")
    #     if len(snip) > max_chars: snip = snip[:max_chars] + " ..."
    #     print(f"[{len(seen)}] arXiv:{m.get('arxiv_id','')}")
    #     print(f"    Title : {title}")
    #     print(f"    Authors: {authors}")
    #     print(f"    Date  : {when}")
    #     print(f"    PDF   : {m.get('pdf_url','')}")
    #     print(f"    Summary  : {snip}\n")

    # doc_order = []
    # doc2chunks = {}

    # for h in hits:
    #     aid = h["metadata"].get("arxiv_id") or h["doc_id"]
    #     if aid not in doc2chunks:
    #         doc2chunks[aid] = []
    #         doc_order.append(aid)
    #     doc2chunks[aid].append({
    #         "chunk_id": h["chunk_id"],
    #         "text": h["text"],
    #     })

    # if save_npz_path:
    #     arxiv_ids = np.array(doc_order, dtype=object)
    #     chunk_ids = np.array([[c["chunk_id"] for c in doc2chunks[aid]] for aid in doc_order], dtype=object)
    #     save_kwargs = {
    #         "arxiv_ids": arxiv_ids,
    #         "chunk_ids": chunk_ids,
    #     }
    #     chunk_texts = np.array([[c["text"] for c in doc2chunks[aid]] for aid in doc_order], dtype=object)
    #     save_kwargs["chunk_texts"] = chunk_texts

    #     np.savez(save_npz_path, **save_kwargs)
    #     print(f"[INFO] Saved mapping to {save_npz_path}. Load with: np.load('{save_npz_path}', allow_pickle=True)")

def pretty_print_docs(hits, max_chars=300, save_npz_path=None, final_answer: str | None = None, question: str | None = None):
    # Group hits by document (best chunk is first for each doc)
    doc_order: list[str] = []
    doc2chunks: dict[str, list[dict]] = {}
    best_per_doc: dict[str, dict] = {}
    meta_per_doc: dict[str, dict] = {}
    for h in hits:
        aid = h["metadata"].get("arxiv_id") or h["doc_id"]
        if aid not in doc2chunks:
            doc2chunks[aid] = []
            doc_order.append(aid)
            best_per_doc[aid] = h  # first seen is the best chunk we appended
            meta_per_doc[aid] = h["metadata"]
        doc2chunks[aid].append(h)

    # Print and collect per-doc summaries
    doc_summaries: list[str] = []
    doc_answers: list[str] = []

    def _format_doc_context(chunks_for_doc, max_ctx_tokens=1800, model_enc="cl100k_base") -> str:
        enc = tiktoken.get_encoding(model_enc)
        parts, used = [], 0
        for it in chunks_for_doc:
            src = f"[Source: arXiv:{it['metadata']['arxiv_id']}]"
            block = f"{src}\n{it['text'].strip()}\n"
            tok = enc.encode(block)
            if used + len(tok) > max_ctx_tokens:
                break
            parts.append(block)
            used += len(tok)
        return "\n---\n".join(parts)

    for idx, aid in enumerate(doc_order, 1):
        m = meta_per_doc[aid]
        title = m.get("title", "") or ""
        authors = ", ".join(m.get("authors", [])) or "Unknown"
        when = m.get("published", "") or ""
        # summarize using all selected chunks for this doc
        summary = summarize_docs(doc2chunks[aid], model=LLM_MODEL) or ""
        snip = (summary.replace("\n", " "))
        if len(snip) > max_chars:
            snip = snip[:max_chars] + " ..."
        print(f"[{idx}] arXiv:{aid}")
        print(f"    Title : {title}")
        print(f"    Authors: {authors}")
        print(f"    Date  : {when}")
        print(f"    PDF   : {m.get('pdf_url','')}")
        print(f"    Summary: {snip}\n")
        doc_summaries.append(summary)

        # optional per-document answer
        if question:
            ctx = _format_doc_context(doc2chunks[aid])
            ans = answer_query_with_context(question, ctx, model=LLM_MODEL)
            doc_answers.append(ans)
        else:
            doc_answers.append("")

    # Save NPZ with extra info
    if save_npz_path:
        arxiv_ids = np.array(doc_order, dtype=object)
        chunk_ids = np.array([[c["chunk_id"] for c in doc2chunks[aid]] for aid in doc_order], dtype=object)
        chunk_texts = np.array([[c["text"] for c in doc2chunks[aid]] for aid in doc_order], dtype=object)
        # doc-level scores from the best chunk
        doc_rel_scores = np.array(
            [float(best_per_doc[aid].get("rel_score", np.nan)) for aid in doc_order],
            dtype="float32",
        )
        doc_recency_scores = np.array(
            [float(best_per_doc[aid].get("recency_score", np.nan)) for aid in doc_order],
            dtype="float32",
        )
        doc_combined_scores = np.array(
            [float(best_per_doc[aid].get("combined_score", np.nan)) for aid in doc_order],
            dtype="float32",
        )
        doc_summaries_arr = np.array(doc_summaries, dtype=object)
        qa_answer_arr = np.array(final_answer or "", dtype=object)  # scalar object array
        doc_answers_arr = np.array(doc_answers, dtype=object)       # one answer per doc

        np.savez(
            save_npz_path,
            arxiv_ids=arxiv_ids,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            doc_rel_scores=doc_rel_scores,
            doc_recency_scores=doc_recency_scores,
            doc_combined_scores=doc_combined_scores,
            doc_summaries=doc_summaries_arr,
            qa_answer=qa_answer_arr,
            doc_answers=doc_answers_arr,
        )
        print(f"[INFO] Saved mapping to {save_npz_path}. Load with: np.load('{save_npz_path}', allow_pickle=True)")

# interest_queries = make_interest_queries(core="translational medicine", focus="edge computing")
hits = retrieve_recent_interest(
    natural_language_query,
    faiss_per_query=80,
    final_docs=12,
    per_doc_chunks=1,
    alpha_recency=0.35
)

# pretty_print_docs(hits, save_npz_path="arxiv_out_hits.npz")
context = format_context(hits)
final_answer = answer_query_with_context(natural_language_query, context, model=LLM_MODEL)
print("\n[ANSWER]\n" + final_answer + "\n")
pretty_print_docs(hits, save_npz_path="arxiv_out_hits.npz", final_answer=final_answer, question=natural_language_query)

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

# def format_context(hits, max_ctx_tokens=1800, model_enc="cl100k_base"):
#     enc = tiktoken.get_encoding(model_enc)
#     parts, used = [], 0
#     for h in hits:
#         src = f"[Source: arXiv:{h['metadata']['arxiv_id']}]"
#         block = f"{src}\n{h['text'].strip()}\n"
#         tok = enc.encode(block)
#         if used + len(tok) > max_ctx_tokens:
#             break
#         parts.append(block)
#         used += len(tok)
#     return "\n---\n".join(parts)

# context = format_context(hits)
# print(context)