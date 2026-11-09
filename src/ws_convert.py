"""Put human demos and official-env agent trajectories into ONE page-visit format and derive
(1) unified sessions for BehaviorGPT scoring (parse.parse_rows, as for the pilot) and
(2) trajectory features computed by the same function for both classes.

Usage: uv run python src/ws_convert.py -> data/processed/ws_sessions.jsonl, ws_traj_features.jsonl
"""

import ast
import glob
import json
import re
import sys
import zipfile
from pathlib import Path
from statistics import mean
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parse import parse_rows  # noqa: E402

P = ROOT / "data/processed"
ASIN = re.compile(r"\b(B[0-9A-Z]{9})\b", re.I)
TOK = re.compile(r"[a-z0-9]+")


def url_to_row(url, obs=None):
    path = url.split("://", 1)[-1].split("/", 1)[-1]  # keep "#" inside queries (not a fragment here)
    parts = [unquote(x) for x in path.split("/")]
    kind = parts[0] if parts and parts[0] in ("search_results", "item_page", "item_sub_page", "done") else "index"
    row = {"page": kind, "url": url, "content": None}
    kw = lambda s: ast.literal_eval(s) if s.startswith("[") else s.replace("+", " ").split()
    if kind == "search_results":
        shown = list(dict.fromkeys(a.upper() for a in ASIN.findall(obs or "")))
        row["content"] = {"keywords": kw(parts[2]), "page": int(parts[3]), "search_result_asins": shown}
    elif kind in ("item_page", "item_sub_page"):
        row["content"] = {"asin": parts[2], "keywords": parts[3], "page": parts[4],
                          "sub_page": parts[5] if kind == "item_sub_page" else None, "options": parts[-1]}
    elif kind == "done":
        row["content"] = {"asin": parts[2], "options": parts[3] if len(parts) > 3 else "{}"}
    return row


def traj_features(rows, instruction):
    """Same definitions for humans and agents; rows = page visits (index excluded from counts)."""
    instr = set(TOK.findall(instruction.lower()))
    pages = [r for r in rows if r["page"] != "index"]
    srch = [r for r in pages if r["page"] == "search_results"]
    queries, last = [], None
    for r in srch:
        k = r["content"]["keywords"]
        q = " ".join(k) if isinstance(k, list) else str(k)
        if q != last:
            queries.append(q)
            last = q
    items = [r["content"]["asin"] for r in pages if r["page"] == "item_page"]
    first_shown = next((r["content"]["search_result_asins"] for r in srch if r["content"]["search_result_asins"]), [])
    bought = next((r["content"]["asin"] for r in pages if r["page"] == "done"), None)
    qt = [set(TOK.findall(q.lower())) for q in queries]
    opt_changes = sum(1 for a, b in zip(pages, pages[1:]) if a["page"] == b["page"] == "item_page"
                      and a["content"]["asin"] == b["content"]["asin"])
    back_to_results = sum(1 for a, b in zip(pages, pages[1:]) if a["page"] in ("item_page", "item_sub_page")
                          and b["page"] == "search_results")
    return {
        "n_pages": len(pages),
        "n_queries": len(queries),
        "n_reformulations": max(len(queries) - 1, 0),
        "query_words": mean(len(q) for q in qt) if qt else 0.0,
        "query_copies_instruction": mean(len(q & instr) / max(len(q), 1) for q in qt) if qt else 0.0,
        "first_query_coverage": len(qt[0] & instr) / max(len(instr), 1) if qt else 0.0,
        "n_item_views": len(set(items)),
        "max_results_page": max((r["content"]["page"] for r in srch), default=0),
        "n_subpages": sum(r["page"] == "item_sub_page" for r in pages),
        "n_option_clicks": opt_changes,
        "n_back_to_results": back_to_results,
        "bought": int(bought is not None),
        "bought_rank": first_shown.index(bought) if bought in first_shown else -1,
        "views_before_buy": len(set(items)) if bought else -1,
    }


def human_rows(sids):
    z = zipfile.ZipFile(ROOT / "data/raw/webshop_human/all_trajs.zip")
    out = {}
    for n in z.namelist():
        stem = Path(n).stem
        if n.endswith(".jsonl") and not n.startswith("__MACOSX") and "h_" + stem in sids:
            out["h_" + stem] = [json.loads(l) for l in z.read(n).decode().splitlines() if l.strip()]
    return out


def main():
    agents = [json.loads(l) for f in sorted(glob.glob(str(P / "ws_traj_*.jsonl"))) for l in open(f)]
    agents = [a for a in agents if "steps" in a]
    pairs = {a["paired_human"] for a in agents}
    humans = {json.loads(l)["sid"]: json.loads(l) for l in open(P / "sessions.jsonl")}
    hrows = human_rows(pairs)
    sess, feats = [], []
    for sid in sorted(pairs):
        h = humans[sid]
        s = dict(h); s["pair"] = sid
        sess.append(s)
        feats.append({"sid": sid, "source": "human", "pair": sid, "reward": float(h.get("reward") or 0),
                      **traj_features(hrows[sid], h["instruction"])})
    for a in agents:
        rows = [url_to_row(st["url"], st["obs"]) for st in a["steps"]]
        u = parse_rows(rows)
        u.update(sid=a["sid"], source=a["source"], instruction=a["instruction"], goal_asin=a["goal_asin"],
                 reward=a["reward"], pair=a["paired_human"])
        sess.append(u)
        feats.append({"sid": a["sid"], "source": a["source"], "pair": a["paired_human"], "reward": float(a["reward"] or 0),
                      **traj_features(rows, a["instruction"])})
    with open(P / "ws_sessions.jsonl", "w") as f:
        for s in sess:
            f.write(json.dumps(s) + "\n")
    with open(P / "ws_traj_features.jsonl", "w") as f:
        for r in feats:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in feats:
        by.setdefault(r["source"], []).append(r)
    for src, rs in by.items():
        print(src, len(rs), {k: round(mean(r[k] for r in rs), 2) for k in
                             ("n_pages", "n_queries", "query_copies_instruction", "n_item_views", "bought", "reward")})


if __name__ == "__main__":
    main()
