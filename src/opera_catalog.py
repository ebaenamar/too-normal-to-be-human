"""Add OPeRA products (every item in a result list or visited page) to the existing catalog.
Only title and price are known for these items. Usage: uv run python src/opera_catalog.py"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
P = ROOT / "data/processed"


def main():
    have = set(json.load(open(P / "catalog_ids.json")))
    rows = [json.loads(l) for l in open(P / "opera_items.jsonl")]
    rows = [r for r in rows if r["id"] not in have and r["name"]]
    n = len(rows)
    print("new items", n, "catalog would be", len(have) + n)
    base = pq.read_table(P / "webshop_pilot.parquet").schema
    t = pa.table({"id": [r["id"] for r in rows], "name": [r["name"][:500] for r in rows], "market": ["us"] * n,
                  "image_url": pa.array([None] * n, type=pa.string()), "brand": pa.array([None] * n, type=pa.string()),
                  "price": [f"{r['price']:.2f}" if r["price"] else None for r in rows],
                  "currency": ["USD"] * n, "search_keywords": pa.array([None] * n, type=pa.list_(pa.string())),
                  "categories": pa.array([None] * n, type=base.field("categories").type),
                  "keywords": pa.array([None] * n, type=pa.list_(pa.string())), "event_type": ["product"] * n,
                  "group": ["product"] * n, "sales_since": pa.array([None] * n, type=pa.list_(pa.int64())),
                  "timestamp": pa.array([None] * n, type=pa.int64()),
                  "frequency": pa.array([None] * n, type=pa.int64())}).select(base.names)
    p = P / "opera_items_add.parquet"
    pq.write_table(t, p)
    from behaviorgpt import UnboxAIClient
    cid = (P / "catalog_id.txt").read_text().strip()
    UnboxAIClient(market="us").add_items(str(p), catalog_id=cid, wait=True)
    json.dump(sorted(have | {r["id"] for r in rows}), open(P / "catalog_ids.json", "w"))
    print("catalog now", len(have) + n)


if __name__ == "__main__":
    main()
