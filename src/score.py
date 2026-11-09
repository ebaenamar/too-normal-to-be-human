"""Score sessions with BehaviorGPT: per-step probability of each choice (vendor method:
head top-100 + id-filtered target and fixed uniform tail sample, softmax with temperature).

For each session, the scored steps are first-time views (and orders of items never viewed).
Context: if the step follows a search, /vector/search with that query and the history before
it; otherwise /vector/recommendations with the history. Responses are cached in SQLite.

Usage: uv run python src/score.py   -> data/processed/obs.jsonl (raw observations per step)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
BASE = "https://behaviorgpt-northeurope.api.unboxai.com/v1"
CAT = (ROOT / "data/processed/catalog_id.txt").read_text().strip()
TZ, DOM = "UTC", {"market": "us"}
HEAD, TAIL = 100, 99
END = datetime(2026, 11, 8, 15, 0, 0, tzinfo=UTC)  # fixed: the model reads y/m/d/weekday/hour

http = httpx.Client(base_url=BASE, headers={"x-api-key": os.environ["UNBOXAI_API_KEY"], "x-catalog-id": CAT},
                    timeout=httpx.Timeout(60.0, connect=10.0))
cache = sqlite3.connect(ROOT / "data/processed/api_cache.sqlite", check_same_thread=False)
cache.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
lock = threading.Lock()


def api_events(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        if e["type"] == "search":
            out.append({"type": "search", "query": e["query"]})
        elif e["type"] == "view":
            out.append({"type": "viewItem", "product": e["item"]})
        elif e["type"] == "order":
            out.append({"type": "addToCart", "product": e["item"]})
            out.append({"type": "order", "cart_id": "c", "items": [{"product": e["item"], "quantity": 1}]})
    t0 = END - timedelta(milliseconds=len(out))
    return [{"timestamp": (t0 + timedelta(milliseconds=i)).isoformat(timespec="milliseconds"),
             "event": ev, "domains": DOM, "timezone": TZ, "store_id": CAT} for i, ev in enumerate(out)]


def post(path: str, body: dict) -> list[tuple[str, float]]:
    body = {"store_id": CAT, "domains": DOM, "timezone": TZ, "record_call": False,
            "fields_to_include": [], **body}
    k = hashlib.sha256((path + json.dumps(body, sort_keys=True)).encode()).hexdigest()
    with lock:
        row = cache.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
    if row:
        return json.loads(row[0])
    for a in range(6):
        r = http.post(path, json=body)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** a, 30)); continue
        if r.status_code >= 400:
            raise RuntimeError(f"{path} {r.status_code} {r.text[:300]}")
        items = [(i["id"], i["score"]) for i in r.json()["products"]["items"]]
        with lock:
            cache.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(items)))
            cache.commit()
        return items
    raise RuntimeError("retries exhausted")


def rank_call(prefix: list[dict], limit: int, filters: dict | None = None):
    """Ranking for the next choice after `prefix` (our events)."""
    body = {"limit": limit}
    if filters:
        body["filters"] = filters
    if prefix and prefix[-1]["type"] == "search":
        body.update(query=prefix[-1]["query"], history=api_events(prefix[:-1]),
                    register_event=False, max_trailing_searches=3)
        return "search", post("/vector/search", body)
    body.update(history=api_events(prefix))
    return "browse", post("/vector/recommendations", body)


def score_session(s: dict, tail_ids: list[str]) -> list[dict]:
    obs, touched = [], set()
    for t, e in enumerate(s["events"]):
        if e["type"] == "search":
            continue
        item = e["item"]
        first = item not in touched
        touched.add(item)
        if not first or t == 0:
            continue
        prefix = s["events"][:t]
        mode, head = rank_call(prefix, HEAD)
        head_ids = {i for i, _ in head}
        tail = [i for i in tail_ids if i not in head_ids][:TAIL]
        _, filt = rank_call(prefix, len(tail) + 1, {"id": tail + [item]})
        fs = dict(filt)
        if item not in fs:
            continue  # item not in catalog
        obs.append({"sid": s["sid"], "source": s["source"], "step": t, "type": e["type"], "mode": mode,
                    "item": item, "pos": e.get("pos"),
                    "head_ids": [i for i, _ in head], "head_scores": [x for _, x in head],
                    "tail_scores": [fs[i] for i in tail if i in fs], "target_score": fs[item]})
    return obs


def main():
    cat_ids = json.load(open(ROOT / "data/processed/catalog_ids.json"))
    tail_ids = random.Random(7).sample(cat_ids, 400)  # fixed uniform sample, reused at every step
    pilot = json.load(open(ROOT / "data/processed/pilot_instructions.json"))
    pilot_h = {p["sid"] for p in pilot[:150]}
    calib_h = {p["sid"] for p in pilot[150:450]}  # humans for temperature/calibration only
    sessions = []
    for line in open(ROOT / "data/processed/sessions.jsonl"):
        s = json.loads(line)
        if s["sid"] in pilot_h:
            s["split"] = "test"; sessions.append(s)
        elif s["sid"] in calib_h:
            s["split"] = "calib"; sessions.append(s)
    for f in (ROOT / "data/processed").glob("agent_*.jsonl"):
        for line in open(f):
            s = json.loads(line); s["split"] = "test"; sessions.append(s)
    print("sessions", len(sessions))
    out = open(ROOT / "data/processed/obs.jsonl", "w")
    n = 0
    with ThreadPoolExecutor(8) as pool:
        for s, obs in zip(sessions, pool.map(lambda s: score_session(s, tail_ids), sessions)):
            out.write(json.dumps({"sid": s["sid"], "source": s["source"], "split": s["split"],
                                  "events": s["events"], "obs": obs}) + "\n")
            n += 1
            if n % 50 == 0:
                print(n, flush=True)
    out.close()
    print("done", n)


if __name__ == "__main__":
    main()
