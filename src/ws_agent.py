"""Full-trajectory LLM agents in the OFFICIAL WebShop text environment (full 1.18M catalog,
original pyserini BM25 index; see ws_build_docs.py / ws_verify.py), on the same goals as the
paired human demonstrations. Every step logs the page URL in the same form as the human logs.

Run with the WebShop venv:
  JAVA_HOME=... vendor/ws-venv/bin/python src/ws_agent.py --model anthropic/claude-haiku-5.5 --n 150
-> data/processed/ws_traj_<slug>.jsonl
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "vendor/WebShop"
RAW = ROOT / "data/raw/webshop_items"
sys.path.insert(0, str(WS))
load_dotenv(ROOT / ".env")
MAX_STEPS = 20
HIST = 4

SYSTEM = """You are shopping on a web store. Follow the instruction to find and buy the right product.
Each turn you see the current page and the available actions. Reply with exactly one action:
search[keywords]   (only when a search bar is available)
click[value]       (value must be one of the clickable values listed)
Format:
Thought: <brief reasoning>
Action: <one action>"""


def make_env():
    import web_agent_site.engine.engine as E
    import web_agent_site.utils as U
    for mod in (E, U):  # full catalog instead of the 1k default
        mod.DEFAULT_ATTR_PATH = str(RAW / "items_ins_v2.json")
        mod.HUMAN_ATTR_PATH = str(RAW / "items_human_ins.json")
    from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv
    server = SimServer("http://127.0.0.1:3000", str(RAW / "items_shuffle.json"), None, -1, None, True, False)
    return server, WebAgentTextEnv


def ask(model, messages):
    for a in range(6):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions", timeout=180,
                           headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                           json={"model": model, "temperature": 0.7, "max_tokens": 3000, "messages": messages})
            if r.status_code == 200:
                c = r.json()["choices"][0]["message"].get("content") or ""
                if c.strip():
                    return c
        except httpx.HTTPError:
            pass
        time.sleep(2 ** a)
    return ""


def parse(text):
    m = re.findall(r"(search|click)\[(.+?)\]", text, flags=re.I | re.S)
    return f"{m[-1][0].lower()}[{m[-1][1].strip()}]" if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    P = ROOT / "data/processed"
    pilot = json.load(open(P / "pilot_instructions.json"))[: args.n]
    want = {p["sid"] for p in pilot}
    humans = {}
    for line in open(P / "sessions.jsonl"):
        s = json.loads(line)
        if s["sid"] in want:
            humans[s["sid"]] = s

    server, Env = make_env()
    norm = lambda t: " ".join(re.sub(r"[^a-z0-9.$ ]", " ", (t or "").lower()).split())
    base = lambda t: norm(re.split(r",? and price lower than", (t or "").lower())[0]).rstrip(".")
    price = lambda t: (lambda m: float(m[1]) if m else 1000000)(re.search(r"price lower than ([0-9.]+) dollars", t.lower()))
    goal_idx = {}
    for i, g in enumerate(server.goals):
        goal_idx.setdefault((g["asin"], base(g["instruction_text"])), i)
    lock = threading.Lock()
    slug = re.sub(r"[^a-z0-9]+", "-", args.model.split("/")[-1].lower())
    out_path = P / f"ws_traj_{slug}.jsonl"
    done = {json.loads(l)["paired_human"] for l in open(out_path)} if out_path.exists() else set()

    def run(h):
        key = (h["goal_asin"], base(h["instruction"]))
        if key not in goal_idx:
            return {"paired_human": h["sid"], "error": "goal not found"}
        with lock:  # same goal as the human, including the human's price limit and wording
            g = server.goals[goal_idx[key]]
            g["price_upper"] = price(h["instruction"])
            g["instruction_text"] = h["instruction"]
            env = Env(observation_mode="text", server=server)
            obs, _ = env.reset(session=goal_idx[key])
            steps = [{"action": None, "url": env.state["url"], "obs": obs}]
        msgs, reward, done_ = [], 0.0, False
        for t in range(MAX_STEPS):
            with lock:
                av = env.get_available_actions()
            acts = ("search[...] or " if av["has_search_bar"] else "") + "click one of: " + ", ".join(av["clickables"])
            user = f"Observation:\n{obs}\n\nAvailable actions: {acts}"
            convo = [{"role": "system", "content": SYSTEM}] + msgs[-2 * HIST:] + [{"role": "user", "content": user}]
            reply = ask(args.model, convo)
            action = parse(reply) or "click[back to search]"
            with lock:
                obs, reward, done_, _ = env.step(action)
                steps.append({"action": action, "url": env.state["url"], "obs": obs})
            msgs += [{"role": "user", "content": user[:3000]}, {"role": "assistant", "content": reply[-1500:]}]
            if done_:
                break
        return {"paired_human": h["sid"], "sid": f"{slug}:{h['sid']}", "source": slug, "model": args.model,
                "instruction": h["instruction"], "goal_asin": h["goal_asin"], "steps": steps,
                "reward": reward, "finished": done_}

    todo = [humans[p["sid"]] for p in pilot if p["sid"] in humans and p["sid"] not in done]
    print("episodes to run", len(todo), flush=True)
    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "a") as f:
        for i, r in enumerate(pool.map(run, todo)):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(i + 1, r.get("error") or f"steps={len(r['steps']) - 1} reward={r['reward']}", flush=True)


if __name__ == "__main__":
    main()
