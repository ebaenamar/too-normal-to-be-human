"""Rebuild the official WebShop search documents (search_engine/convert_product_file_format.py
+ engine.load_products), byte-for-byte in the indexed `contents` and in document order, so a
pyserini 0.17 BM25 index over them reproduces the engine the WebShop humans used.
The stored `product` blob is omitted: only `contents` is searched and `id` is read back.

Usage: uv run python src/ws_build_docs.py -> vendor/WebShop/search_engine/resources/documents.jsonl
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/webshop_items/items_shuffle.json"
OUT = ROOT / "vendor/WebShop/search_engine/resources/documents.jsonl"


def main():
    products = json.load(open(SRC))
    print("products", len(products))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(OUT, "w") as f:
        for p in products:
            asin = p["asin"]
            if asin == "nan" or len(asin) > 10 or asin in seen:
                continue
            seen.add(asin)
            bullets = p["small_description"] if isinstance(p["small_description"], list) else [p["small_description"]]
            options = {}
            for name, contents in (p["customization_options"] or {}).items():
                if contents is None:
                    continue
                options[name.lower()] = [c["value"].strip().replace("/", " | ").lower() for c in contents]
            option_text = ", and ".join(f"{k}: {', '.join(v)}" for k, v in options.items())
            contents = " ".join([p["name"], p["full_description"], bullets[0], option_text]).lower()
            f.write(json.dumps({"id": asin, "contents": contents}) + "\n")
            n += 1
    print("documents", n)


if __name__ == "__main__":
    main()
