"""Build and upload the pilot catalog (<= 100k items, the per-catalog limit).

Contents: every item any human or pilot agent touched or was shown, plus a seeded uniform
fill from the 1.18M WebShop products up to --size. The same catalog serves both classes,
so neither is scored against a store the other never saw.

Usage: uv run python src/build_catalog.py --size 60000 [--upload]
"""

import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def session_files():
    return [ROOT / "data/processed/sessions.jsonl"] + [Path(p) for p in glob.glob(str(ROOT / "data/processed/agent_*.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=60000)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    need = set()
    for f in session_files():
        for line in open(f):
            s = json.loads(line)
            if s["source"] == "agent_traj":
                continue  # 1000-product world, excluded (catalog confound)
            for e in s["events"]:
                if e["type"] == "search":
                    need.update(e.get("shown") or [])
                else:
                    need.add(e["item"])

    rows, ids = {}, []
    with open(ROOT / "data/processed/items.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            if r["id"] in need:
                rows[r["id"]] = r
    print(f"needed {len(need)}, found {len(rows)}")
    rng = random.Random(0)
    pool = [i for i in ids if i not in rows]
    fill_ids = set(rng.sample(pool, max(0, args.size - len(rows))))
    with open(ROOT / "data/processed/items.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in fill_ids:
                rows[r["id"]] = r

    recs = list(rows.values())
    table = pa.table({
        "id": [r["id"] for r in recs],
        "name": [r["name"] for r in recs],
        "market": ["us"] * len(recs),
        "image_url": [r.get("image_url") for r in recs],
        "brand": [r.get("brand") for r in recs],
        "price": [r.get("price") for r in recs],
        "currency": ["USD"] * len(recs),
        "search_keywords": pa.array([None] * len(recs), type=pa.list_(pa.string())),
        "categories": [r.get("categories") for r in recs],
        "keywords": pa.array([None] * len(recs), type=pa.list_(pa.string())),
        "event_type": ["product"] * len(recs),
        "group": ["product"] * len(recs),
        "sales_since": pa.array([None] * len(recs), type=pa.list_(pa.int64())),
        "timestamp": pa.array([None] * len(recs), type=pa.int64()),
        "frequency": pa.array([None] * len(recs), type=pa.int64()),
    })
    out = ROOT / "data/processed/webshop_pilot.parquet"
    pq.write_table(table, out)
    json.dump(sorted(rows), open(ROOT / "data/processed/catalog_ids.json", "w"))
    print("wrote", out, len(recs), "items")

    if args.upload:
        from behaviorgpt import UnboxAIClient
        client = UnboxAIClient(market="us")
        job = client.embed(str(out), wait=True, timeout=3600.0)
        print("catalog_id", job.catalog_id)
        (ROOT / "data/processed/catalog_id.txt").write_text(job.catalog_id)


if __name__ == "__main__":
    main()
