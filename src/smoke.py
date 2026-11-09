"""Gate-1 smoke test of the BehaviorGPT ranking API, on the public `retail_catalog`.

Checks, each printed as PASS/FAIL with numbers, all saved to results/smoke.json:
  1. determinism   - the same request three times returns the same ids and scores
  2. id filter     - filters={"id": [...]} returns exactly those items, with the same
                     scores they had in the unfiltered ranking (so a chosen item can be scored)
  3. history       - the history changes the ranking (vs an empty history)
  4. timing-free   - the same history, events spaced 1 ms vs ~20 min apart within the
                     same hour, gives the same scores (the model reads only y/m/d/weekday/hour)
  5. throughput    - calls per minute and 429s with 1, 4 and 8 parallel workers

Never records events: register_event=False on every call.
Usage:  uv run python src/smoke.py            (needs UNBOXAI_API_KEY in .env)
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
KEY = os.environ.get("UNBOXAI_API_KEY", "").strip()
BASE = "https://behaviorgpt-northeurope.api.unboxai.com/v1"
CATALOG = os.environ.get("SMOKE_CATALOG", "retail_catalog")
TZ, DOM = "UTC", {"market": "us"}

if not KEY:
    sys.exit("UNBOXAI_API_KEY is empty: put it in .env (never in code or chat).")

http = httpx.Client(
    base_url=BASE,
    headers={"x-api-key": KEY, "x-catalog-id": CATALOG},
    timeout=httpx.Timeout(60.0, connect=10.0),
)
stats = {"calls": 0, "429": 0, "errors": 0}


def ev(e: dict, t: datetime) -> dict:
    return {"timestamp": t.isoformat(timespec="milliseconds"), "event": e,
            "domains": DOM, "timezone": TZ, "store_id": CATALOG}


def hist(events: list[dict], spacing: timedelta = timedelta(milliseconds=1),
         end: datetime | None = None) -> list[dict]:
    end = end or (datetime.now(tz=UTC) - timedelta(seconds=5))
    t0 = end - spacing * (len(events) - 1)
    return [ev(e, t0 + spacing * i) for i, e in enumerate(events)]


def call(path: str, body: dict, retries: int = 5) -> dict:
    body = {"store_id": CATALOG, "domains": DOM, "timezone": TZ, "record_call": False, **body}
    for a in range(retries + 1):
        stats["calls"] += 1
        r = http.post(path, json=body)
        if r.status_code == 429:
            stats["429"] += 1
            time.sleep(min(2 ** a, 30))
            continue
        if r.status_code >= 400:
            stats["errors"] += 1
            raise RuntimeError(f"{path} {r.status_code}: {r.text[:500]}")
        return r.json()
    raise RuntimeError("429 after retries")


def rec(history: list[dict], limit=100, filters=None, personalized=True) -> list[tuple[str, float]]:
    body = {"history": history, "limit": limit, "personalized": personalized}
    if filters:
        body["filters"] = filters
    items = call("/vector/recommendations", body)["products"]["items"]
    return [(i["id"], i.get("score")) for i in items]


def search(query: str, history: list[dict], limit=100, filters=None) -> list[tuple[str, float]]:
    body = {"query": query, "history": history, "limit": limit, "register_event": False,
            "max_trailing_searches": 3}
    if filters:
        body["filters"] = filters
    items = call("/vector/search", body)["products"]["items"]
    return [(i["id"], i.get("score")) for i in items]


def main() -> None:
    out: dict = {"catalog": CATALOG, "when": datetime.now(tz=UTC).isoformat()}

    # seed items from the catalog
    seed = search("running shoes", [], limit=50)
    seed2 = search("coffee maker", [], limit=50)
    assert seed and seed2, "empty search results"
    a, b = seed[3][0], seed2[2][0]
    out["seed_null_scores"] = sum(s is None for _, s in seed + seed2)

    H = [{"type": "search", "query": "running shoes"}, {"type": "viewItem", "product": a}]

    # 1 determinism
    runs = [rec(hist(H)) for _ in range(3)]
    same_ids = all([i for i, _ in r] == [i for i, _ in runs[0]] for r in runs)
    max_dscore = max(abs(x[1] - y[1]) for r in runs[1:] for x, y in zip(r, runs[0]))
    out["1_determinism"] = {"pass": same_ids and max_dscore < 1e-6, "same_ids": same_ids,
                            "max_score_diff": max_dscore}

    # 2 id filter: score items deep in the ranking and one far outside it
    head = dict(runs[0])
    probe = [runs[0][0][0], runs[0][50][0], runs[0][99][0], b]
    filt = rec(hist(H), limit=len(probe), filters={"id": probe})
    got = dict(filt)
    consistent = all(abs(got[i] - head[i]) < 1e-5 for i in probe if i in head and i in got)
    out["2_id_filter"] = {"pass": set(got) == set(probe) and consistent,
                          "returned": len(got), "asked": len(probe),
                          "consistent_with_head": consistent,
                          "outside_item_score": got.get(b), "head_top_score": runs[0][0][1],
                          "head_100th_score": runs[0][99][1]}

    # 3 history matters
    empty = rec([])
    unpers = rec(hist(H), personalized=False)
    top = lambda r, k=20: {i for i, _ in r[:k]}
    out["3_history"] = {
        "pass": len(top(runs[0]) & top(empty)) < 15,
        "top20_overlap_vs_empty": len(top(runs[0]) & top(empty)),
        "top20_overlap_vs_personalized_false": len(top(runs[0]) & top(unpers)),
    }

    # 4 timing-free: same hour, 1 ms vs 20 min spacing
    end = datetime.now(tz=UTC).replace(minute=59, second=0, microsecond=0) - timedelta(hours=1)
    H3 = H + [{"type": "viewItem", "product": seed[5][0]}]
    fast = rec(hist(H3, timedelta(milliseconds=1), end))
    slow = rec(hist(H3, timedelta(minutes=20), end))
    d = max(abs(x[1] - y[1]) for x, y in zip(fast, slow))
    out["4_timing_free"] = {"pass": [i for i, _ in fast] == [i for i, _ in slow] and d < 1e-6,
                            "max_score_diff": d}

    # 5 throughput
    tp = {}
    for w in (1, 4, 8):
        n, before429 = 24, stats["429"]
        t0 = time.time()
        with ThreadPoolExecutor(w) as pool:
            list(pool.map(lambda k: rec(hist(H), limit=100), range(n)))
        dt = time.time() - t0
        tp[w] = {"calls_per_min": round(60 * n / dt, 1), "sec_per_call": round(dt / n, 3),
                 "new_429": stats["429"] - before429}
    out["5_throughput"] = tp
    out["stats"] = stats

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/smoke.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if isinstance(v, dict) and "pass" in v:
            print(("PASS " if v["pass"] else "FAIL ") + k, {x: y for x, y in v.items() if x != "pass"})
    print("throughput", tp, "stats", stats)


if __name__ == "__main__":
    main()
