"""Session-level human vs agent detection in the official WebShop environment (same engine).

BehaviorGPT session features (vendor method; tau, loc, scale fitted on human sessions of the
OTHER fold, 2-fold by instruction pair): mean excess surprisal u, z = sum(u)/sqrt(n), mean
log-rank, one-sided CUSUM of u ("surprising") and of -u ("too normal"), k = 0.5.
Trajectory features: ws_convert.traj_features. Detectors: single-feature AUROC (agent positive)
and logistic regressions with 5-fold CV grouped by instruction pair; 95% CIs by pair bootstrap.

Usage: uv run python src/ws_analyze.py -> results/ws_trajectory.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor/behaviorgpt-anomaly"))
from anomaly.scoring import Observation, fit_tau, predict, robust_location_scale  # noqa: E402
from opera_analyze import boot_auc, cv_auc  # noqa: E402

P = ROOT / "data/processed"
EPS, K = 0.02, 0.5
TRAJ = ["n_pages", "n_queries", "n_reformulations", "query_words", "query_copies_instruction",
        "first_query_coverage", "n_item_views", "max_results_page", "n_subpages", "n_option_clicks",
        "n_back_to_results", "bought", "bought_rank", "views_before_buy"]
BG = ["bgpt_mean_u", "bgpt_z", "bgpt_logrank", "bgpt_cusum_surprise", "bgpt_cusum_normal"]


def cusum(x):
    s, m = 0.0, 0.0
    for v in x:
        s = max(0.0, s + v - K); m = max(m, s)
    return m


def main():
    n_items = len(json.load(open(P / "catalog_ids.json")))
    O = {o["sid"]: o for o in map(json.loads, open(P / "ws_obs.jsonl"))}
    F = {r["sid"]: r for r in map(json.loads, open(P / "ws_traj_features.jsonl"))}
    pairs = sorted({r["pair"] for r in F.values()})
    fold = {p: i % 2 for i, p in enumerate(pairs)}
    obs = lambda o: Observation(mode=o["mode"], head_ids=o["head_ids"], head_scores=o["head_scores"],
                                tail_scores=o["tail_scores"], target_score=o["target_score"], n_items=n_items)
    cal = {}
    for k in (0, 1):
        hs = [o for s in O.values() if s["source"] == "human" and fold[s["pair"]] != k for o in s["obs"]]
        cal[k] = {}
        for m in ("search", "browse"):
            om = [obs(o) for o in hs if o["mode"] == m]
            tau = fit_tau(om, EPS) if len(om) > 5 else 0.02
            ex = [(lambda p: p.surprisal - p.entropy)(predict(o, tau, EPS)) for o in om]
            loc, sc = robust_location_scale(ex) if len(ex) > 5 else (0.0, 1.0)
            cal[k][m] = (tau, loc, sc)
    for sid, r in F.items():
        us, lr = [], []
        for o in O.get(sid, {"obs": []})["obs"]:
            tau, loc, sc = cal[fold[r["pair"]]][o["mode"]]
            p = predict(obs(o), tau, EPS)
            us.append(float(np.clip(((p.surprisal - p.entropy) - loc) / sc, -3, 6))); lr.append(math.log(p.rank))
        r["n_scored"] = len(us)
        r["bgpt_mean_u"] = float(np.mean(us)) if us else np.nan
        r["bgpt_z"] = sum(us) / math.sqrt(len(us)) if us else np.nan
        r["bgpt_logrank"] = float(np.mean(lr)) if lr else np.nan
        r["bgpt_cusum_surprise"] = cusum(us) if us else np.nan
        r["bgpt_cusum_normal"] = cusum([-u for u in us]) if us else np.nan

    report = {"n_pairs": len(pairs), "conditions": {}}
    agents = sorted({r["source"] for r in F.values()} - {"human"})
    for a in agents:
        have = {r["pair"] for r in F.values() if r["source"] == a}
        sub = [r for r in F.values() if r["pair"] in have and r["source"] in ("human", a)]
        y = np.array([int(r["source"] != "human") for r in sub]); g = np.array([r["pair"] for r in sub])
        rep = {"n_sessions": {"human": int((y == 0).sum()), "agent": int((y == 1).sum())},
               "median_scored_steps": {"human": float(np.median([r["n_scored"] for r in sub if r["source"] == "human"])),
                                       "agent": float(np.median([r["n_scored"] for r in sub if r["source"] == a]))},
               "features": {}, "detectors": {}}
        print(f"\n=== human vs {a}: {rep['n_sessions']}, scored steps {rep['median_scored_steps']}")
        print(f"{'feature':26s} {'human':>7s} {'agent':>7s}  AUROC [95% CI]")
        for k in TRAJ + BG + ["reward"]:
            x = np.array([r[k] for r in sub], float)
            if np.all(np.isnan(x)) or len(set(x[~np.isnan(x)])) < 2:
                continue
            auc, lo, hi = boot_auc(y, x, g)
            hm, am = np.nanmean(x[y == 0]), np.nanmean(x[y == 1])
            print(f"{k:26s} {hm:7.2f} {am:7.2f}  {auc:.3f} [{lo:.3f},{hi:.3f}]")
            rep["features"][k] = {"human_mean": hm, "agent_mean": am, "auroc": [auc, lo, hi]}
        for name, cols in [("Trajectory features", TRAJ), ("BGPT only", BG), ("Trajectory + BGPT", TRAJ + BG)]:
            X = np.array([[r[c] for c in cols] for r in sub], float)
            med = np.nanmedian(X, axis=0)
            X = np.where(np.isnan(X), med, X)  # sessions with no scored step: column median
            pr = cv_auc(X, y, g)
            auc, lo, hi = boot_auc(y, pr, g)
            print(f"LR {name:24s} AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]")
            rep["detectors"][name] = [auc, lo, hi]
        report["conditions"][a] = rep
    json.dump(report, open(ROOT / "results/ws_trajectory.json", "w"), indent=1, default=float)
    with open(P / "ws_session_table.jsonl", "w") as f:
        for r in F.values():
            f.write(json.dumps(r, default=float) + "\n")


if __name__ == "__main__":
    main()
