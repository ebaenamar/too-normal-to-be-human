"""Pilot analysis: BehaviorGPT surprisal statistics vs cheap baselines, humans vs agents.

Temperature and location/scale are fitted on calibration humans only (vendor method).
AUROC: agent = positive class. AUROC < 0.5 means agents score LOWER than humans
("too normal"); |AUROC - 0.5| is the separability. 95% CIs by bootstrap over sessions.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/behaviorgpt-anomaly"))
from anomaly.scoring import Observation, fit_tau, predict, robust_location_scale  # noqa: E402

EPS = 0.02
N_ITEMS = len(json.load(open(ROOT / "data/processed/catalog_ids.json")))


def to_obs(o):
    return Observation(mode=o["mode"], head_ids=o["head_ids"], head_scores=o["head_scores"],
                       tail_scores=o["tail_scores"], target_score=o["target_score"], n_items=N_ITEMS)


def auc_ci(y, x, B=1000, seed=0):
    y, x = np.asarray(y), np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    if len(set(y)) < 2:
        return float("nan"), (float("nan"), float("nan")), len(y)
    a = roc_auc_score(y, x)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], x[i]))
    return a, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), len(y)


def main():
    S = [json.loads(l) for l in open(ROOT / "data/processed/obs.jsonl")]
    calib = [o for s in S if s["split"] == "calib" for o in s["obs"]]
    tau, loc, scale = {}, {}, {}
    for m in ("search", "browse"):
        obs_m = [to_obs(o) for o in calib if o["mode"] == m]
        tau[m] = fit_tau(obs_m, EPS) if obs_m else 0.01
        ex = [(lambda p: p.surprisal - p.entropy)(predict(o, tau[m], EPS)) for o in obs_m]
        loc[m], scale[m] = robust_location_scale(ex) if ex else (0.0, 1.0)
    print("tau", tau, "n calib steps", {m: sum(o["mode"] == m for o in calib) for m in tau})

    rows = []
    for s in S:
        if s["split"] != "test":
            continue
        us, lr, sur = [], [], []
        for o in s["obs"]:
            p = predict(to_obs(o), tau[o["mode"]], EPS)
            u = ((p.surprisal - p.entropy) - loc[o["mode"]]) / scale[o["mode"]]
            us.append(float(np.clip(u, -3, 6))); lr.append(math.log(p.rank)); sur.append(p.surprisal)
        ev = s["events"]
        views = [e for e in ev if e["type"] == "view"]
        pos = [e["pos"] for e in views if e.get("pos") is not None]
        qs = [e["query"] for e in ev if e["type"] == "search"]
        rows.append({
            "sid": s["sid"], "source": s["source"], "agent": int(s["source"] != "human"),
            "n_scored": len(us),
            "bgpt_z": sum(us) / math.sqrt(len(us)) if us else np.nan,
            "bgpt_mean_u": float(np.mean(us)) if us else np.nan,
            "bgpt_max_u": float(np.max(us)) if us else np.nan,
            "bgpt_mean_logrank": float(np.mean(lr)) if lr else np.nan,
            "bgpt_mean_surprisal": float(np.mean(sur)) if sur else np.nan,
            "B1_mean_pos": float(np.mean(pos)) if pos else np.nan,
            "B1_frac_top1": float(np.mean([p == 0 for p in pos])) if pos else np.nan,
            "B2_n_events": len(ev),
            "B2_n_searches": len(qs),
            "B2_n_views": len(views),
            "B3_query_words": float(np.mean([len(q.split()) for q in qs])) if qs else np.nan,
        })
    feats = [k for k in rows[0] if k.startswith(("bgpt", "B"))]
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
    print({k: len(v) for k, v in groups.items()})
    print("scored steps per session (median):", {k: float(np.median([r["n_scored"] for r in v])) for k, v in groups.items()})

    report = {"tau": tau, "n_items": N_ITEMS, "groups": {k: len(v) for k, v in groups.items()}, "auroc": {}}
    agents = [k for k in groups if k != "human"]
    for cmp in ["all_agents"] + agents:
        sub = [r for r in rows if r["source"] == "human" or (r["source"] == cmp or cmp == "all_agents")]
        y = [r["agent"] for r in sub]
        print(f"\n== humans vs {cmp}  (n={len(sub)})")
        print(f"{'feature':24s} {'AUROC':>6s}  95% CI          sep   human_med  agent_med")
        report["auroc"][cmp] = {}
        for f in feats:
            a, (lo, hi), n = auc_ci(y, [r[f] for r in sub])
            hm = np.nanmedian([r[f] for r in sub if not r["agent"]])
            am = np.nanmedian([r[f] for r in sub if r["agent"]])
            print(f"{f:24s} {a:6.3f}  [{lo:.3f},{hi:.3f}]  {abs(a - .5):.3f}  {hm:9.3f}  {am:9.3f}")
            report["auroc"][cmp][f] = {"auroc": a, "ci": [lo, hi], "n": n, "human_median": hm, "agent_median": am}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(report, open(ROOT / "results/pilot_auroc.json", "w"), indent=1, default=float)
    json.dump(rows, open(ROOT / "results/pilot_sessions.json", "w"), default=float)


if __name__ == "__main__":
    main()
