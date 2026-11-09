"""Parse WebShop human demos and AgentTraj-L agent trajectories into one session format.

Session = {
  "sid": str, "source": "human" | "agent_traj",
  "instruction": str,           # normalized instruction text (used for grouped splits)
  "events": [                   # in order
     {"type": "search", "query": str, "shown": [asin, ...], "page": int}
     {"type": "view", "item": asin, "pos": int | None}   # pos = 0-based position in last shown list
     {"type": "order", "item": asin}
  ],
  "n_raw_actions": int,
}
Option clicks (size/colour), sub-pages (description/reviews) and back/next navigation are
not events the model reads; they are counted in n_raw_actions only.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ASIN_RE = re.compile(r"^B[0-9A-Z]{9}$")


def norm_instr(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^.*?instruction:\s*\[sep\]\s*", "", s)
    s = s.split("[sep]")[0]
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- humans
def parse_human(path: str) -> dict | None:
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return None
    goal = rows[0].get("goal") or {}
    events, shown, nraw = [], [], 0
    last_kw, cur_item, seen = None, None, set()
    for r in rows:
        p = r["page"]
        c = r.get("content") or {}
        if p == "search_results":
            kw = c.get("keywords")
            q = " ".join(k.replace('"', "") for k in kw) if isinstance(kw, list) else str(kw)
            page = int(c.get("page", 1))
            shown = list(c.get("search_result_asins") or [])
            nraw += 1
            if q != last_kw:
                events.append({"type": "search", "query": q, "shown": shown, "page": page})
                last_kw = q
            else:  # pagination or back-to-results: extend what was shown, no new search
                if events and events[-1]["type"] == "search" and events[-1]["query"] == q:
                    events[-1]["shown"] = events[-1]["shown"] + [a for a in shown if a not in events[-1]["shown"]]
        elif p == "item_page":
            nraw += 1
            a = r.get("asin") or c.get("asin") or _asin_from_url(r.get("url", ""))
            if a and a != cur_item:
                pos = shown.index(a) if a in shown else None
                if a not in seen:
                    events.append({"type": "view", "item": a, "pos": pos})
                    seen.add(a)
                cur_item = a
        elif p == "item_sub_page":
            nraw += 1
        elif p == "done":
            nraw += 1
            a = c.get("asin") or r.get("asin") or cur_item
            if a:
                events.append({"type": "order", "item": a})
    return {
        "sid": "h_" + Path(path).stem,
        "source": "human",
        "instruction": norm_instr(goal.get("instruction_text", "")),
        "goal_asin": goal.get("asin"),
        "events": events,
        "n_raw_actions": nraw,
        "reward": next((r.get("reward") for r in rows if r["page"] == "done"), None),
    }


def _asin_from_url(u: str) -> str | None:
    for part in unquote(u).split("/"):
        if ASIN_RE.match(part):
            return part
    return None


# ---------------------------------------------------------------- agents
ACT_RE = re.compile(r"Action:\s*(search|click)\[(.*?)\]\s*$", re.S | re.I)


def parse_obs_results(obs: str) -> list[tuple[str, str, str | None]]:
    """(asin, name, price) triples from a results-page observation."""
    parts = [p.strip() for p in obs.split("[SEP]")]
    out = []
    for i, p in enumerate(parts):
        if ASIN_RE.match(p) and i + 1 < len(parts):
            price = parts[i + 2] if i + 2 < len(parts) and parts[i + 2].startswith("$") else None
            out.append((p, parts[i + 1], price))
    return out


def parse_agent(rec: dict, idx: int, names: dict) -> dict | None:
    conv = rec["conversations"]
    instr, events, shown, nraw = "", [], [], 0
    cur_item, seen, last_q = None, set(), None
    for t in conv:
        v = t.get("value") or ""
        if t.get("from") == "human":
            if "Instruction:" in v and not instr:
                instr = norm_instr(v)
            res = parse_obs_results(v)
            if res:
                for a, n, pr in res:
                    names.setdefault(a, (n, pr))
                page_asins = [a for a, _, _ in res]
                if events and events[-1]["type"] == "search" and not events[-1]["shown"]:
                    events[-1]["shown"] = page_asins
                elif events and events[-1]["type"] == "search":
                    events[-1]["shown"] += [a for a in page_asins if a not in events[-1]["shown"]]
                shown = events[-1]["shown"] if events and events[-1]["type"] == "search" else page_asins
            continue
        m = ACT_RE.search(v.strip())
        if not m:
            continue
        kind, arg = m.group(1).lower(), m.group(2).strip()
        nraw += 1
        if kind == "search":
            if arg != last_q:
                events.append({"type": "search", "query": arg, "shown": [], "page": 1})
                last_q = arg
        else:
            a = arg.upper()
            if ASIN_RE.match(a):
                if a != cur_item and a not in seen:
                    pos = shown.index(a) if a in shown else None
                    events.append({"type": "view", "item": a, "pos": pos})
                    seen.add(a)
                cur_item = a
            elif arg.lower() == "buy now" and cur_item:
                events.append({"type": "order", "item": cur_item})
    return {
        "sid": f"a_{idx}",
        "source": "agent_traj",
        "instruction": instr,
        "events": events,
        "n_raw_actions": nraw,
        "reward": None,
    }


def main(root: str) -> None:
    root = Path(root)
    humans = [s for f in sorted(glob.glob(str(root / "data/raw/webshop_human/ext/all_trajs/*.jsonl")))
              if (s := parse_human(f))]
    names: dict = {}
    agents = [s for i, r in enumerate(json.load(open(root / "data/raw/agenttraj/webshop_train.json")))
              if (s := parse_agent(r, i, names))]
    out = root / "data/processed"
    with open(out / "sessions.jsonl", "w") as f:
        for s in humans + agents:
            f.write(json.dumps(s) + "\n")
    json.dump(names, open(out / "agent_obs_names.json", "w"))
    print(f"humans={len(humans)} agents={len(agents)} names_from_agent_obs={len(names)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
