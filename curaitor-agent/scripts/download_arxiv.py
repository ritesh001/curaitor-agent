import arxiv
from pathlib import Path
import json, re, argparse, os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Optional LLM providers
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
RAW_ROOT = DATA_DIR / "raw"
RAW_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_ALLOWED_CATS = [
    "cond-mat.mtrl-sci",
    "physics.chem-ph",
    "cond-mat.soft",
    "physics.app-ph",
    "cs.MS",
]

def slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-zA-Z0-9\-_ ]+", "", s)
    s = s.replace(" ", "_")
    return s[:max_len] or "query"

def load_config(path: Path | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    import yaml
    return yaml.safe_load(Path(path).read_text())

def get_llm(config: dict):
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "gemini").lower()
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
            raise RuntimeError("pip install langchain-google-genai")
        api_key = llm_cfg.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or llm.api_key in config.yaml")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

def rewrite_query_with_llm(user_instruction: str, config: dict, allowed_cats: list[str]) -> str:
    """
    Convert a verbose instruction into a compact arXiv query string.
    Output must be a single line arXiv search query using AND/OR, quoted phrases, and cat: filters.
    """
    llm = get_llm(config)
    cats = " OR ".join(f"cat:{c}" for c in allowed_cats)
    prompt = f"""You turn a user's research instruction into an arXiv API search query.
Rules:
- Output ONLY the query string (one line), no commentary.
- Use quoted phrases for key terms.
- Combine terms with AND/OR.
- Include category filters: ({cats})
- Prefer title/abstract fields: ti:, abs:
- Do NOT include the user's formatting/extraction instructions.

User instruction:
{user_instruction}

Examples:
Instruction: "Find recent papers on liquid electrolytes, extract ionic conductivity."
Query: (ti:"liquid electrolyte" OR abs:"liquid electrolyte") AND (ti:"ionic conductivity" OR abs:"ionic conductivity") AND ({cats})

Now produce the query:"""
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        return text.strip().splitlines()[0]
    except Exception:
        # Heuristic fallback
        words = [w for w in re.findall(r"[A-Za-z0-9\-]+", user_instruction.lower()) if len(w) > 3]
        uniq = []
        for w in words:
            if w not in uniq:
                uniq.append(w)
        base = " AND ".join(f'(ti:"{w}" OR abs:"{w}")' for w in uniq[:6])
        return f"({base}) AND ({cats})"

def download_arxiv_pdfs(user_instruction: str, max_results: int = 12, out_dir: Path | None = None, config_path: Path | None = None, use_llm: bool = False, allowed_cats: list[str] | None = None):
    config = load_config(config_path)
    allowed_cats = allowed_cats or DEFAULT_ALLOWED_CATS
    query = rewrite_query_with_llm(user_instruction, config, allowed_cats) if use_llm else user_instruction

    slug = slugify(query)
    out = (out_dir or RAW_ROOT / slug)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"

    print(f"[arXiv] User instruction: {user_instruction}")
    print(f"[arXiv] Search query: {query}")
    print(f"[arXiv] Output dir: {out}")

    client = arxiv.Client(page_size=25, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = list(client.results(search))
    # Filter by allowed categories (if provided)
    filtered = []
    for r in results:
        cats = list(getattr(r, "categories", []))
        if not cats or any(c in cats for c in allowed_cats):
            filtered.append(r)

    records = []
    for r in tqdm(filtered, desc="Downloading PDFs"):
        arxiv_id = r.entry_id.split("/")[-1]
        title = r.title
        pdf_name = f"{arxiv_id}_{slugify(title)}.pdf"
        pdf_path = out / pdf_name
        if not pdf_path.exists():
            try:
                r.download_pdf(dirpath=out, filename=pdf_name)
            except Exception as e:
                print(f"[WARN] Failed to download {arxiv_id}: {e}")
                continue
        records.append({
            "instruction": user_instruction,
            "query": query,
            "query_slug": slug,
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
            "published": r.published.strftime("%Y-%m-%d"),
            "pdf_path": str(pdf_path),
            "primary_category": getattr(r, "primary_category", None),
            "categories": list(getattr(r, "categories", [])),
        })

    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[arXiv] Wrote manifest with {len(records)} entries: {manifest}")
    return manifest

def main():
    ap = argparse.ArgumentParser(description="Download arXiv PDFs for a query")
    ap.add_argument("query", nargs="+", help="user instruction (will be LLM-rewritten if --use-llm)")
    ap.add_argument("--max-results", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="path to config.yaml for LLM credentials")
    ap.add_argument("--use-llm", action="store_true", help="rewrite query with LLM before searching")
    ap.add_argument("--cats", nargs="*", default=None, help="allowed categories (override)")
    args = ap.parse_args()

    user_instruction = " ".join(args.query)
    allowed_cats = args.cats or DEFAULT_ALLOWED_CATS
    download_arxiv_pdfs(
        user_instruction,
        max_results=args.max_results,
        out_dir=args.out_dir,
        config_path=args.config,
        use_llm=args.use_llm,
        allowed_cats=allowed_cats,
    )

if __name__ == "__main__":
    main()