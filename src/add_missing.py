"""Add items touched or shown in agent sessions that the uploaded catalog lacks."""
import glob, json
from pathlib import Path
import pyarrow.parquet as pq, pyarrow as pa
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]; load_dotenv(ROOT / ".env")
have = set(json.load(open(ROOT / "data/processed/catalog_ids.json")))
need = set()
for f in glob.glob(str(ROOT / "data/processed/agent_*.jsonl")):
    for l in open(f):
        for e in json.loads(l)["events"]:
            need.update(e.get("shown") or []) if e["type"] == "search" else need.add(e["item"])
miss = need - have
rows = [json.loads(l) for l in open(ROOT / "data/processed/items.jsonl")]
rows = [r for r in rows if r["id"] in miss]
print("missing", len(miss), "found", len(rows))
base = pq.read_table(ROOT / "data/processed/webshop_pilot.parquet").schema
n = len(rows)
t = pa.table({"id": [r["id"] for r in rows], "name": [r["name"] for r in rows], "market": ["us"]*n,
    "image_url": [r.get("image_url") for r in rows], "brand": [r.get("brand") for r in rows],
    "price": [r.get("price") for r in rows], "currency": ["USD"]*n,
    "search_keywords": pa.array([None]*n, type=pa.list_(pa.string())), "categories": [r.get("categories") for r in rows],
    "keywords": pa.array([None]*n, type=pa.list_(pa.string())), "event_type": ["product"]*n, "group": ["product"]*n,
    "sales_since": pa.array([None]*n, type=pa.list_(pa.int64())), "timestamp": pa.array([None]*n, type=pa.int64()),
    "frequency": pa.array([None]*n, type=pa.int64())}).select(base.names)
p = ROOT / "data/processed/agent_items_add.parquet"; pq.write_table(t, p)
from behaviorgpt import UnboxAIClient
cid = (ROOT / "data/processed/catalog_id.txt").read_text().strip()
UnboxAIClient(market="us").add_items(str(p), catalog_id=cid, wait=True)
json.dump(sorted(have | {r["id"] for r in rows}), open(ROOT / "data/processed/catalog_ids.json", "w"))
print("catalog now", len(have | {r['id'] for r in rows}))
