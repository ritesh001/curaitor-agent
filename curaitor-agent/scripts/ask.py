import sys, json, yaml, os, argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_ROOT = DATA_DIR / "index"
CONFIG = BASE_DIR / "config.yaml"

def load_corpus(corpus_path: Path):
    if not corpus_path.exists():
        print(f"No index found at {corpus_path}.")
        sys.exit(1)
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def find_latest_corpus() -> Path | None:
    corpora = sorted(INDEX_ROOT.glob("*/corpus.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return corpora[0] if corpora else None

def tokenize(s: str):
    return [t for t in s.lower().split() if t.isascii()]

def score_chunks(query: str, chunks):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scored = []
    for rec in chunks:
        text = rec["text"].lower()
        tokens = tokenize(text)
        tf = sum(tokens.count(t) for t in q_tokens)
        bonus = 3 if all(t in text for t in q_tokens[: min(3, len(q_tokens))]) else 0
        title_bonus = 1 if q_tokens[0] in rec["title"].lower() else 0
        score = tf + bonus + title_bonus
        if score > 0:
            scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for _, rec in scored]

def get_llm(config):
    llm_cfg = config.get("llm", {})
    # provider = llm_cfg.get("provider", "gemini").lower()
    provider = llm_cfg.get("provider", "openrouter").lower()
    model = llm_cfg.get("model")
    temperature = llm_cfg.get("temperature", 0)

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError("pip install langchain-ollama")
        return ChatOllama(model=model, temperature=temperature, base_url=llm_cfg.get("base_url", "http://localhost:11434"))
    elif provider == "openrouter":
        if ChatOpenAI is None:
            raise RuntimeError("pip install langchain-openai")
        api_key = llm_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY or llm.api_key in config.yaml")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
        )
    else:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("pip install langchain-google-genai python-dotenv")
        api_key = llm_cfg.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or llm.api_key in config.yaml")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

def build_prompt(query: str, contexts):
    ctx_texts, citations = [], []
    for i, c in enumerate(contexts, 1):
        ctx_texts.append(f"[{i}] Title: {c['title']} (doc_id={c['doc_id']}, chunk={c['chunk_id']})\n{c['text']}")
        citations.append(f"[{i}] {c['title']} (arXiv:{c['doc_id']})")
    return f"""You are a helpful research assistant. Answer strictly using the provided context. If not in context, say you don't know.

Question:
{query}

Context:
{'\n\n'.join(ctx_texts)}

Instructions:
- Cite sources inline like [1], [2].
- Be concise.

Available sources:
{'\n'.join(citations)}
"""

def main():
    ap = argparse.ArgumentParser(description="Ask a question over the latest per-query corpus")
    ap.add_argument("query", nargs="+", help="question to answer")
    ap.add_argument("--corpus", type=Path, default=None, help="path to corpus.jsonl")
    args = ap.parse_args()

    user_query = " ".join(args.query)
    corpus_path = args.corpus or find_latest_corpus()
    if not corpus_path:
        print("No corpus found. Run extract_and_index.py first.")
        sys.exit(1)

    chunks = load_corpus(corpus_path)
    ranked = score_chunks(user_query, chunks)
    top_k = ranked[:8] if ranked else []
    if not top_k:
        print("No relevant context found in index.")
        sys.exit(0)

    config = {}
    if CONFIG.exists():
        config = yaml.safe_load(CONFIG.read_text())
    llm = get_llm(config)
    prompt = build_prompt(user_query, top_k)

    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        print("\n=== Answer ===\n")
        print(text.strip())
        print("\n=== Sources ===")
        for i, c in enumerate(top_k, 1):
            print(f"[{i}] {c['title']} (arXiv:{c['doc_id']}) — {c['pdf_path']}")
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()