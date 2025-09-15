from datetime import datetime, time, timedelta
from dateutil import tz
import arxiv
import tiktoken
import re, requests
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sentence_transformers import CrossEncoder
from collections import defaultdict
from datetime import datetime
from dateutil import tz
import numpy as np
import tiktoken

## TODO: combine into one query => use LLM to find keywords: Yuxing
#arXiv 的查询语法（字段如 all:, ti:, au:, abs:, cat:，支持 AND/OR/括号与引号等）。
QUERY = 'all:"digital phenotyping"' 
# Step 6) 定义你的研究兴趣查询集合（可自由增删）
def make_interest_queries(core="digital phenotyping", focus="mental health"):
    q = [
        core,
        f"{core} {focus}",
        "review survey digital phenotyping"
    ]
    # 去重清洗
    seen, out = set(), []
    for s in q:
        s = " ".join(s.split()).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower()); out.append(s)
    return out

# 伦敦时区 & 时间窗：过去10天（含今天）
tz_london = tz.gettz("Europe/London")
now_local = datetime.now(tz_london)
days = 100
start_date = (now_local.date() - timedelta(days=days-1))  # 含今天共100天
start_dt = datetime.combine(start_date, time(0, 0, tzinfo=tz_london))
end_dt   = datetime.combine(now_local.date(), time(23, 59, 59, tzinfo=tz_london))

# 是否按“最后更新(updated)”筛选；默认 False=按首次提交(published)
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

def _strip_hyphenation(t: str) -> str:
    return re.sub(r"-\s*\n\s*", "", t)

def _normalize_ws(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)  # 多个空行压成一个
    paras = [re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in t.split("\n\n")]
    paras = [re.sub(r"\s{2,}", " ", p) for p in paras if p]
    return "\n\n".join(paras).strip()

def _cut_refs(t: str) -> str:
    patt = re.compile(r"\n\s*(references|bibliography|acknowledg(e)?ments)\s*\n", re.I)
    last = None
    for m in patt.finditer("\n"+t+"\n"):
        last = m
    return t[:last.start()].strip() if last else t

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
## TODO: save chunk => Ritesh
enc = tiktoken.get_encoding("cl100k_base")

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


EMB_MODEL = "BAAI/bge-small-en-v1.5"  # 多语可换 "BAAI/bge-m3" ## TODO: test different embedding models & save embeddings => Ritesh
emb = SentenceTransformer(EMB_MODEL)

chunk_texts = [c["text"] for c in chunks]
X = emb.encode(chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
X = np.asarray(X, dtype="float32")
print("[INFO] Embeddings:", X.shape)



#⑤ 建索引（FAISS 内积；等价于余弦，因为已归一化）
# --- Step 5: FAISS index ---

dim = X.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X)
print("[INFO] Index size:", index.ntotal)



#⑥ 检索 + 交叉编码器复排（MiniLM）
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
def pretty_print_docs(hits, max_chars=300):
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
        print(f"    Snip  : {snip}\n") ## TODO: show more context => Ritesh

# === 用法示例 ===
interest_queries = make_interest_queries(core="translational medicine", focus="edge computing")
hits = retrieve_recent_interest(
    interest_queries,
    faiss_per_query=80,
    final_docs=12,
    per_doc_chunks=1,   # 每篇1段，便于LLM摘要
    alpha_recency=0.35  # 越大越偏新
)
pretty_print_docs(hits)

#⑦ 组包
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