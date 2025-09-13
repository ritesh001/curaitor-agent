from .llm_agent import ExtractionAgent
import yaml
import json
import os
from rag.content_parsing import extract_pdf_components
from crawler.arxiv_downloader import download_arxiv_pdfs

class AgentPipeline:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.agent = ExtractionAgent(self.config)

    def _load_items(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().strip()
            if not data:
                return []
            return json.loads(data)

    def run(self, crawled_items_path='crawled.json', output_path=None):
        # Step 1: Download PDFs from arXiv
        download_arxiv_pdfs(self.config['arxiv']['download_path'])

        # Step 2: Load crawled items
        items = self._load_items(crawled_items_path)
        if not items:
            print("No crawled items.")
            return
        out = output_path or self.config['output']['path']
        os.makedirs(os.path.dirname(out), exist_ok=True)

        all_results = []
        for idx, item in enumerate(items):
            try:
                if 'file_path' in item and os.path.exists(item['file_path']):
                    text = extract_pdf_components(item['file_path'])['texts']
                else:
                    text = "\n".join(
                        str(item.get(k,'')) for k in ['title','abstract']
                        if item.get(k)
                    )
                if not text.strip():
                    continue
                res = self.agent.extract(text, self.config['extraction']['schema'])
                all_results.append({
                    "source_index": idx,
                    "metadata": {k: item.get(k) for k in ['title','authors'] if k in item},
                    "extraction": res
                })
            except Exception as e:
                print(f"[WARN] Item {idx} failed: {e}")

        with open(out, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote {len(all_results)} records to {out}")