"""Check that the rebuilt WebShop index reproduces what the human demonstrators saw: re-run every
human search (lower-cased, top 50, page 1 = ranks 1-10) and compare with the logged first page.

Run with the WebShop venv (pyserini 0.17, Java 11):
  JAVA_HOME=... vendor/ws-venv/bin/python src/ws_verify.py
"""

import json
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

ROOT = Path(__file__).resolve().parents[1]


def main():
    s = LuceneSearcher(str(ROOT / "vendor/WebShop/search_engine/indexes"))
    pairs = []
    for line in open(ROOT / "data/processed/sessions.jsonl"):
        r = json.loads(line)
        if r["source"] != "human":
            continue
        for e in r["events"]:
            if e["type"] == "search" and e.get("page", 1) == 1 and len(e["shown"]) >= 10:
                pairs.append((e["query"], e["shown"][:10]))
    exact, overlap, top1 = 0, 0.0, 0
    for q, shown in pairs:
        hits = s.search(" ".join(q.lower().split(" ")), k=50)
        got = [h.docid for h in hits][:10]
        exact += got == shown
        overlap += len(set(got) & set(shown)) / 10
        top1 += bool(got) and got[0] == shown[0]
    n = len(pairs)
    out = {"searches": n, "exact_first_page": exact / n, "mean_overlap_at_10": overlap / n, "top1_match": top1 / n}
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "results/ws_engine_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
