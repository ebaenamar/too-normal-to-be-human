"""BehaviorGPT scoring of whole WebShop sessions (humans + official-env agents), vendor method as
in score.py. Items the agents saw that the catalog lacks are added first.

Usage: uv run python src/ws_score.py -> data/processed/ws_obs.jsonl
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score import score_session  # noqa: E402

P = ROOT / "data/processed"


def add_missing(sessions):
    have = set(json.load(open(P / "catalog_ids.json")))
    need = set()
    for s in sessions:
        for e in s["events"]:
            need.update(e.get("shown") or []) if e["type"] == "search" else need.add(e["item"])
    miss = need - have
    if not miss:
        return
    rows = [r for r in map(json.loads, open(P / "items.jsonl")) if r["id"] in miss]
    print("adding", len(rows), "of", len(miss), "missing items")
    if not rows:
        return
    base = pq.read_table(P / "webshop_pilot.parquet")
    t = pa.Table.from_pylist([{**{k: None for k in base.schema.names}, "id": r["id"], "name": r["name"], "market": "us",
                               "image_url": r.get("image_url"), "brand": r.get("brand"), "price": r.get("price"),
                               "currency": "USD", "categories": r.get("categories"), "event_type": "product",
                               "group": "product"} for r in rows], schema=base.schema)
    p = P / "ws_items_add.parquet"
    pq.write_table(t, p)
    from behaviorgpt import UnboxAIClient
    UnboxAIClient(market="us").add_items(str(p), catalog_id=(P / "catalog_id.txt").read_text().strip(), wait=True)
    json.dump(sorted(have | {r["id"] for r in rows}), open(P / "catalog_ids.json", "w"))


def main():
    sessions = [json.loads(l) for l in open(P / "ws_sessions.jsonl")]
    add_missing(sessions)
    cat_ids = json.load(open(P / "catalog_ids.json"))
    tail_ids = json.load(open(P / "tail_ids.json")) if (P / "tail_ids.json").exists() else None
    if tail_ids is None:  # the pilot's fixed tail sample, drawn from the pilot catalog
        pilot_cat = sorted(set(cat_ids))
        tail_ids = random.Random(7).sample(pilot_cat, 400)
        json.dump(tail_ids, open(P / "tail_ids.json", "w"))
    with ThreadPoolExecutor(8) as pool, open(P / "ws_obs.jsonl", "w") as f:
        for s, obs in zip(sessions, pool.map(lambda s: score_session(s, tail_ids), sessions)):
            f.write(json.dumps({"sid": s["sid"], "source": s["source"], "pair": s["pair"], "obs": obs}) + "\n")
    print("scored", len(sessions))


if __name__ == "__main__":
    main()
