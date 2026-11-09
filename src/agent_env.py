"""A text WebShop-style environment over the full 1.18M-product WebShop catalog, and an
LLM agent (via OpenRouter) that shops in it for the same instructions the humans had.

Pilot caveat: search is SQLite FTS5 BM25 over name/category/description, an approximation
of WebShop's Pyserini BM25 engine that the humans used. Option pages (size/colour) are not
modelled: options are not events BehaviorGPT reads.

Usage: uv run python src/agent_env.py --model anthropic/claude-haiku-5.5 --n 100
Writes data/processed/agent_<model>.jsonl in the session format of parse.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
PAGE = 10
MAX_STEPS = 15

SYSTEM = """You are web shopping.
I will give you instructions about what to do. You have to follow the instructions.
Every round I will give you an observation and a list of available actions; respond with one action.
You can use search action if search is available. You can click one of the buttons in clickables.
An action should be of the following structure:
search[keywords]
click[value]
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Your response should use the following format:

Thought:
I think ...

Action:
click[something]"""


class Shop:
    def __init__(self):
        import threading
        self.db = sqlite3.connect(ROOT / "data/processed/items.sqlite", check_same_thread=False)
        self.lock = threading.Lock()
        self.items = {}
        meta = {}
        with open(ROOT / "data/processed/items.jsonl") as f:
            for line in f:
                r = json.loads(line)
                meta[r["id"]] = (r["name"], r.get("categories") or "", "")
        self.meta = meta

    def item(self, asin):
        return self.meta.get(asin, ("(unknown item)", "", ""))

    def search(self, q: str, k: int = 50) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", q.lower())
        if not toks:
            return []
        expr = " OR ".join(f'"{t}"' for t in toks[:30])
        with self.lock:
            rows = self.db.execute(
                "SELECT asin FROM fts WHERE fts MATCH ? ORDER BY bm25(fts, 0, 3.0, 1.0, 1.0) LIMIT ?", (expr, k)
            ).fetchall()
        return [r[0] for r in rows]


PRICE_RE = re.compile(r"\$?([0-9][0-9,]*\.[0-9]{2})")


def run_episode(shop: Shop, instruction: str, model: str, sid: str, seed: int, prices: dict) -> dict:
    key = os.environ["OPENROUTER_API_KEY"]
    msgs = [{"role": "system", "content": SYSTEM}]
    events, results, page, cur, seen, last_q = [], [], 0, None, set(), None
    state = "index"
    n_raw, usage, transcript = 0, {"in": 0, "out": 0}, []

    def obs_text():
        head = f"Instruction: [SEP] {instruction}"
        if state == "index":
            return head + " [SEP] Search", ["search"]
        if state == "results":
            sl = results[page * PAGE:(page + 1) * PAGE]
            parts = [head, "Back to Search", f"Page {page + 1} (Total results: {len(results)})"]
            clicks = ["back to search"]
            if page > 0:
                parts.append("< Prev"); clicks.append("< prev")
            if (page + 1) * PAGE < len(results):
                parts.append("Next >"); clicks.append("next >")
            for a in sl:
                name, _, _ = shop.item(a)
                parts += [a, name, f"${prices.get(a) or 'N.A.'}"]
                clicks.append(a.lower())
            return " [SEP] ".join(parts), clicks
        name, cat, desc = shop.item(cur)
        parts = [head, "Back to Search", "< Prev", name, f"Price: ${prices.get(cur) or 'N.A.'}",
                 f"Category: {cat}", f"Description: {desc[:300]}", "Buy Now"]
        return " [SEP] ".join(parts), ["back to search", "< prev", "buy now"]

    for step in range(MAX_STEPS):
        o, clicks = obs_text()
        msgs.append({"role": "user", "content": f"Observation:\n{o}\n\nAvailable Actions:\n" +
                     json.dumps({"has_search_bar": state == "index", "clickables": clicks})})
        for attempt in range(5):
            try:
                r = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                               headers={"Authorization": f"Bearer {key}"},
                               json={"model": model, "messages": msgs, "temperature": 0.7,
                                     "max_tokens": 400, "seed": seed}, timeout=90)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt); continue
                r.raise_for_status()
                d = r.json()
                text = d["choices"][0]["message"]["content"] or ""
                u = d.get("usage") or {}
                usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
                break
            except (httpx.HTTPError, KeyError):
                time.sleep(2 ** attempt)
        else:
            break
        msgs.append({"role": "assistant", "content": text})
        transcript.append(text[-300:])
        m = re.findall(r"(search|click)\[(.*?)\]", text, re.I)
        if not m:
            continue
        kind, arg = m[-1][0].lower(), m[-1][1].strip()
        n_raw += 1
        if kind == "search" and state == "index":
            results, page, state = shop.search(arg), 0, "results"
            if arg != last_q:
                events.append({"type": "search", "query": arg, "shown": results[:PAGE], "page": 1})
                last_q = arg
        elif kind == "click":
            a = arg.lower()
            if a not in clicks:
                continue
            if a == "back to search":
                state = "index"
            elif a == "next >":
                page += 1
                if events and events[-1]["type"] == "search":
                    events[-1]["shown"] += [x for x in results[page * PAGE:(page + 1) * PAGE]]
            elif a == "< prev":
                if state == "item":
                    state = "results"
                else:
                    page = max(0, page - 1)
            elif a == "buy now" and cur:
                events.append({"type": "order", "item": cur})
                break
            elif state == "results":
                asin = arg.upper()
                cur, state = asin, "item"
                if asin not in seen:
                    shown = results[:(page + 1) * PAGE]
                    events.append({"type": "view", "item": asin, "pos": shown.index(asin) if asin in shown else None})
                    seen.add(asin)
    return {"sid": sid, "source": f"agent:{model}", "instruction": instruction, "events": events,
            "n_raw_actions": n_raw, "reward": None, "usage": usage, "transcript_tail": transcript[-3:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    shop = Shop()
    prices = {}
    with open(ROOT / "data/processed/items.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("price"):
                prices[r["id"]] = r["price"]
    sess = [json.loads(l) for l in open(ROOT / "data/processed/sessions.jsonl")]
    pilot = json.load(open(ROOT / "data/processed/pilot_instructions.json"))[: args.n]
    slug = re.sub(r"[^a-z0-9]+", "_", args.model.lower())
    out = ROOT / f"data/processed/agent_{slug}.jsonl"
    done = {json.loads(l)["sid"] for l in open(out)} if out.exists() else set()
    todo = [(p, f"g_{slug}_{i}") for i, p in enumerate(pilot) if f"g_{slug}_{i}" not in done]
    with open(out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for s in pool.map(lambda x: run_episode(shop, x[0]["instruction"], args.model, x[1], 0, prices), todo):
            s["paired_human"] = next(p["sid"] for p in pilot if p["instruction"] == s["instruction"])
            f.write(json.dumps(s) + "\n"); f.flush()
    print("wrote", out)


if __name__ == "__main__":
    main()
