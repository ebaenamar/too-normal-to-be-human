"""Engine-controlled statistic: model preference for the chosen item WITHIN the result list
that was actually shown (cancels which search engine produced the list).
Per search step: c = log softmax_tau(chosen | shown) + log(n_shown)  (0 = chance, >0 = model-preferred)
and the within-list rank percentile of the chosen item.
"""
import json, math, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import score  # reuses cached API calls
from analyze import auc_ci
ROOT = Path(__file__).resolve().parents[1]
TAU = json.load(open(ROOT / "results/pilot_auroc.json"))["tau"]["search"]
S = [json.loads(l) for l in open(ROOT / "data/processed/obs.jsonl")]

def one(s):
    out = []
    for o in s["obs"]:
        if o["mode"] != "search":
            continue
        q = s["events"][o["step"] - 1]
        shown = list(dict.fromkeys((q.get("shown") or [])[:20] + [o["item"]]))
        if len(shown) < 3:
            continue
        _, r = score.rank_call(s["events"][: o["step"]], len(shown), {"id": shown})
        sc = dict(r)
        if o["item"] not in sc:
            continue
        v = np.array([sc[i] for i in shown if i in sc]) / TAU
        lse = v.max() + math.log(np.exp(v - v.max()).sum())
        c = sc[o["item"]] / TAU - lse + math.log(len(v))
        pct = float(np.mean(v < sc[o["item"]] / TAU))
        out.append((c, pct))
    return out

with ThreadPoolExecutor(8) as pool:
    res = list(pool.map(one, S))
rows = []
for s, r in zip(S, res):
    if s["split"] != "test" or not r:
        continue
    rows.append({"sid": s["sid"], "src": s["source"], "agent": int(s["source"] != "human"),
                 "c": float(np.mean([x for x, _ in r])), "pct": float(np.mean([p for _, p in r]))})
json.dump(rows, open(ROOT / "results/pilot_choiceset.json", "w"))
for cmp in ["all", "haiku", "deepseek"]:
    sub = [x for x in rows if not x["agent"] or cmp == "all" or cmp in x["src"]]
    for f in ["c", "pct"]:
        a, ci, n = auc_ci([x["agent"] for x in sub], [x[f] for x in sub])
        hm = np.median([x[f] for x in sub if not x["agent"]]); am = np.median([x[f] for x in sub if x["agent"]])
        print(f"{cmp:9s} {f:4s} AUROC={a:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}] n={n} human_med={hm:.3f} agent_med={am:.3f}")
# paired
A = {json.loads(l)["sid"]: json.loads(l)["paired_human"] for f in (ROOT / "data/processed").glob("agent_*.jsonl") for l in open(f)}
by = {x["sid"]: x for x in rows}
from scipy.stats import wilcoxon
for f in ["c", "pct"]:
    d = np.array([by[a][f] - by[h][f] for a, h in A.items() if a in by and h in by])
    print(f"paired {f}: n={len(d)} median(agent-human)={np.median(d):+.3f} frac agent>human={np.mean(d>0):.2f} p={wilcoxon(d).pvalue:.4f}")
