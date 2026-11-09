"""Counterfactual replay: at every human click on an Amazon result list (OPeRA), an LLM agent
acting for that user (their self-reported persona, the same recent history, the same query and
the identical result list) picks the product to open. Engine and intent are held fixed by
construction; only the chooser differs.

Usage: uv run python src/opera_agent.py --model anthropic/claude-haiku-5.5 [--no-persona]
-> data/processed/opera_agent_<slug>[_np].jsonl  ({step_idx, choice, pos, raw})
"""

import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
P = ROOT / "data/processed"
HIST = 12  # recent stream events shown to the agent


def personas():
    u = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ROOT / "data/raw/opera/OPeRA_full/user/*/*.parquet"))])
    out = {}
    for r in u.itertuples():
        try:
            s = json.loads(r.survey)
        except Exception:
            continue
        out[r.user_id] = json.dumps({k: s[k] for k in ("Demographic Information", "Self Description",
                                                       "Shopping Preference") if k in s})[:3000]
    return out


def describe(e, names):
    if e["type"] == "search":
        return f'searched "{e["query"]}"'
    n = names.get(e["item"], "a product")
    return ("viewed " if e["type"] == "view" else "added to cart ") + f'"{n[:90]}"'


def prompt(step, stream, names, persona):
    hist = [describe(e, names) for e in stream[: step["t"] - 1][-HIST:]]
    lines = [f'[{i + 1}] {it["name"][:150]} | ' + (f'${it["price"]:.2f}' if it["price"] else "price n/a") +
             (f' | {it["rating"]} stars' if it.get("rating") else "") + (" | Sponsored" if it.get("sponsored") else "")
             for i, it in enumerate(step["shown"])]
    who = f"You are shopping on Amazon on behalf of this person, as they would:\n{persona}\n\n" if persona else \
        "You are shopping on Amazon.\n\n"
    return (who + "Recent activity (oldest first):\n" + ("\n".join("- " + h for h in hist) or "- (none)") +
            f'\n\nYou just searched "{step["query"]}". The results page shows, in this order:\n' + "\n".join(lines) +
            "\n\nWhich product do you open next? Reply with only the number in brackets, e.g. 3.")


def ask(model, text):
    for a in range(6):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions", timeout=120,
                           headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                           json={"model": model, "temperature": 0.7, "max_tokens": 3000,
                                 "messages": [{"role": "user", "content": text}]})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
        except httpx.HTTPError:
            pass
        time.sleep(2 ** a)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-persona", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    steps = [json.loads(l) for l in open(P / "opera_steps.jsonl")]
    streams = json.load(open(P / "opera_streams.json"))
    names = {json.loads(l)["id"]: json.loads(l)["name"] for l in open(P / "opera_items.jsonl")}
    pers = {} if args.no_persona else personas()

    def run(i):
        s = steps[i]
        raw = ask(args.model, prompt(s, streams[s["user"]], names, pers.get(s["user"])))
        m = re.search(r"\d+", raw)
        k = int(m[0]) - 1 if m else -1
        ok = 0 <= k < len(s["shown"])
        return {"step_idx": i, "choice": s["shown"][k]["id"] if ok else None, "pos": k if ok else None, "raw": raw[:40]}

    slug = re.sub(r"[^a-z0-9]+", "-", args.model.split("/")[-1].lower()) + ("_np" if args.no_persona else "")
    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(run, range(len(steps))))
    with open(P / f"opera_agent_{slug}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    ok = [o for o in out if o["choice"]]
    agree = sum(o["choice"] == steps[o["step_idx"]]["chosen"] for o in ok)
    print(slug, "valid", len(ok), "/", len(out), "agree with human", agree,
          "median pos", pd.Series([o["pos"] for o in ok]).median(), "human median pos",
          pd.Series([s["pos"] for s in steps]).median(), "personas", len(pers))


if __name__ == "__main__":
    main()
