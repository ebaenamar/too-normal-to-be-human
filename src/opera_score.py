"""Score every OPeRA replay step with BehaviorGPT: one /vector/search call with the user's real
history and the query, filtered to the exact result list, gives a score for every shown item.
Human and agent choices are then compared on the same scores. A second call with empty
history is the no-history control.

Usage: uv run python src/opera_score.py -> data/processed/opera_scores.jsonl
"""

import json
import sys
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score import CAT, DOM, TZ, post  # noqa: E402  (same client, cache and catalog)

P = ROOT / "data/processed"
MAXH = 100


def api_history(events, known):
    out = []
    for e in events[-MAXH:]:
        if e["type"] == "search":
            ev = {"type": "search", "query": e["query"]}
        elif e["item"] in known:
            ev = {"type": "viewItem" if e["type"] == "view" else "addToCart", "product": e["item"]}
        else:
            continue
        ts = datetime.fromisoformat(e["ts"]).astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        out.append({"timestamp": ts, "event": ev, "domains": DOM,
                    "timezone": TZ, "store_id": CAT})
    return out


def main():
    known = set(json.load(open(P / "catalog_ids.json")))
    steps = [json.loads(l) for l in open(P / "opera_steps.jsonl")]
    streams = json.load(open(P / "opera_streams.json"))

    def run(i):
        s = steps[i]
        ids = [it["id"] for it in s["shown"] if it["id"] in known]
        hist = api_history(streams[s["user"]][: s["t"] - 1], known)
        body = {"query": s["query"], "limit": len(ids), "filters": {"id": ids}, "register_event": False,
                "max_trailing_searches": 3}
        withh = dict(post("/vector/search", {**body, "history": hist}))
        noh = dict(post("/vector/search", {**body, "history": []}))
        return {"step_idx": i, "n_hist": len(hist), "scores": withh, "scores_nohist": noh}

    with ThreadPoolExecutor(8) as pool, open(P / "opera_scores.jsonl", "w") as f:
        for n, o in enumerate(pool.map(run, range(len(steps)))):
            f.write(json.dumps(o) + "\n")
    print("scored", len(steps))


if __name__ == "__main__":
    main()
