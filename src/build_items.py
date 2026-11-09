"""Stream WebShop's items_shuffle.json (5.5 GB, 1.18M products) into
  data/processed/items.jsonl   one compact row per product (catalog fields)
  data/processed/items.sqlite  an FTS5 BM25 index over name + category + short description,
                               the search engine the pilot agents use
"""

import ast
import json
import re
import sqlite3
import sys
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/webshop_items/items_shuffle.json"
OUT = ROOT / "data/processed"


def first_image(v):
    if isinstance(v, list):
        return v[0] if v else None
    if isinstance(v, str) and v.startswith("["):
        try:
            l = ast.literal_eval(v)
            return l[0] if l else None
        except Exception:
            return None
    return v or None


def clean_brand(b):
    if not b:
        return None
    return re.sub(r"^(Brand:\s*|Visit the\s+)", "", str(b)).replace(" Store", "").strip() or None


def text_of(v, n=400):
    if isinstance(v, list):
        v = " ".join(map(str, v))
    return str(v or "")[:n]


def main():
    db = sqlite3.connect(OUT / "items.sqlite")
    db.execute("DROP TABLE IF EXISTS fts")
    db.execute("CREATE VIRTUAL TABLE fts USING fts5(asin UNINDEXED, name, cat, desc, tokenize='porter')")
    n = 0
    with open(SRC, "rb") as f, open(OUT / "items.jsonl", "w") as out:
        batch = []
        for it in ijson.items(f, "item"):
            asin = it.get("asin")
            name = (it.get("name") or "").strip()
            if not asin or not name:
                continue
            cat = (it.get("product_category") or "").replace("›", ",")
            cat = re.sub(r"\s*,\s*", ", ", cat).strip(", ")
            price = it.get("pricing") or ""
            m = re.search(r"\$?([0-9][0-9,]*\.?[0-9]*)", str(price))
            row = {
                "id": asin, "name": name[:300], "brand": clean_brand(it.get("brand")),
                "categories": cat or None, "image_url": first_image(it.get("images")),
                "price": m.group(1).replace(",", "") if m else None,
            }
            out.write(json.dumps(row) + "\n")
            batch.append((asin, name, cat, text_of(it.get("small_description"))))
            n += 1
            if len(batch) >= 20000:
                db.executemany("INSERT INTO fts VALUES (?,?,?,?)", batch)
                db.commit()
                batch.clear()
                print(n, flush=True)
        if batch:
            db.executemany("INSERT INTO fts VALUES (?,?,?,?)", batch)
            db.commit()
    db.execute("INSERT INTO fts(fts) VALUES('optimize')")
    db.commit()
    print("done", n)


if __name__ == "__main__":
    sys.exit(main())
