"""Parse OPeRA (full) into per-user event streams and replayable click steps.

Stream events (all sessions of a user, in time order): search{query}, view{item},
addToCart{item}. A step is a human click on a product in an Amazon search-result list:
{user, sid, t (index in stream; the prefix is stream[:t]), query, shown [asin...], chosen, pos}.
Items: every product seen in a result list or visited, with title and price.

Usage: uv run python src/opera_parse.py -> data/processed/opera_{streams,steps,items}.json*
"""

import glob
import html as H
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/opera/OPeRA_full"
OUT = ROOT / "data/processed"
ASIN = re.compile(r"(?:/dp/|%2Fdp%2F|/gp/product/)([A-Z0-9]{10})")
PRICE = re.compile(r"\$([0-9][0-9,]*\.[0-9]{2})")
TAG = re.compile(r"<[^>]+>")


def text(s):
    return " ".join(H.unescape(TAG.sub(" ", s)).split())


def parse_results(h: str):
    """Organic and sponsored result cards on an Amazon search page, in page order."""
    out, seen = [], set()
    for b in h.split('class="search-result"')[1:]:
        b = b.split('class="search-result"')[0][:20000]
        m = ASIN.search(b)
        t = re.search(r'class="product-name">(.*?)</a>', b, re.S)
        if not m or not t or m[1] in seen:
            continue
        seen.add(m[1])
        p = PRICE.search(b)
        r = re.search(r"([0-9.]+) out of 5 stars", b)
        out.append({"id": m[1], "name": text(t[1]), "price": float(p[1].replace(",", "")) if p else None,
                    "rating": float(r[1]) if r else None, "sponsored": "sspa" in b[:400]})
    return out


def search_query(url: str):
    u = urlparse(url or "")
    if not u.path.startswith("/s"):
        return None
    k = parse_qs(u.query.replace("&URL_MASKED", "")).get("k")
    return " ".join(unquote(k[0]).split()).lower() if k else None


def main():
    cols = ["session_id", "timestamp", "action_type", "navigation_type", "url", "click_type", "semantic_id",
            "element_meta", "simplified_html"]
    a = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(RAW / "action/*.parquet")))])
    a["user"] = a.session_id.str.split("_").str[0]
    a["ts"] = pd.to_datetime(a.timestamp, utc=True, format="ISO8601")
    a = a.sort_values(["user", "ts"], kind="stable")

    items, streams, steps = {}, {}, []
    for user, g in a.groupby("user", sort=False):
        ev, cur_q, cur_page = [], None, None
        for r in g.itertuples():
            url = r.url or ""
            if r.action_type == "navigation" and r.navigation_type != "reload":
                q = search_query(url)
                m = ASIN.search(url)
                if q:
                    if not (ev and ev[-1]["type"] == "search" and ev[-1]["query"] == q):
                        ev.append({"type": "search", "query": q, "ts": r.ts.isoformat(), "sid": r.session_id})
                    cur_q, cur_page = q, None
                elif m:
                    cur_page = m[1]
                    if not (ev and ev[-1].get("item") == cur_page and ev[-1]["type"] == "view"):
                        ev.append({"type": "view", "item": cur_page, "ts": r.ts.isoformat(), "sid": r.session_id})
                    h = r.simplified_html if isinstance(r.simplified_html, str) else ""
                    t = re.search(r"<title>(.*?)</title>", h, re.S)
                    if t and cur_page not in items:
                        name = re.sub(r"^Amazon\.com\s*:?\s*", "", text(t[1])).rsplit(" : ", 1)[0]
                        if name:
                            items[cur_page] = {"id": cur_page, "name": name, "price": None}
            elif r.action_type == "click" and r.click_type == "purchase" and cur_page and \
                    re.search(r"add_to_cart|buy_now|set_up_now", r.semantic_id or ""):
                ev.append({"type": "addToCart", "item": cur_page, "ts": r.ts.isoformat(), "sid": r.session_id})
            elif r.action_type == "click" and r.click_type == "product_link" and search_query(url):
                try:
                    d = json.loads(json.loads(r.element_meta)["data"])
                except Exception:
                    continue
                res = parse_results(r.simplified_html or "")
                for it in res:
                    items.setdefault(it["id"], {k: it[k] for k in ("id", "name", "price")})
                ids = [it["id"] for it in res]
                chosen = d.get("asin")
                q = search_query(url)
                if chosen in ids and len(ids) >= 5:
                    if not (ev and ev[-1]["type"] == "search" and ev[-1]["query"] == q):
                        # back-navigation to a results page is not always logged: re-enter the search
                        ev.append({"type": "search", "query": q, "ts": r.ts.isoformat(), "sid": r.session_id})
                    steps.append({"user": user, "sid": r.session_id, "t": len(ev), "query": q,
                                  "shown": res, "chosen": chosen, "pos": ids.index(chosen)})
        streams[user] = ev

    for s in streams.values():
        for e in s:
            if e["type"] != "search" and e["item"] not in items:
                e["unknown"] = True  # no title captured; kept in the stream, dropped from API history
    json.dump(streams, open(OUT / "opera_streams.json", "w"))
    with open(OUT / "opera_steps.jsonl", "w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")
    with open(OUT / "opera_items.jsonl", "w") as f:
        for it in items.values():
            f.write(json.dumps(it) + "\n")
    lens = pd.Series([len(v) for v in streams.values()])
    print("users", len(streams), "stream events", lens.describe().round(1).to_dict())
    print("steps", len(steps), "users with steps", len({s['user'] for s in steps}),
          "list size", pd.Series([len(s["shown"]) for s in steps]).describe().round(1).to_dict())
    print("history length at step", pd.Series([s["t"] for s in steps]).describe().round(1).to_dict())
    print("items", len(items))


if __name__ == "__main__":
    main()
