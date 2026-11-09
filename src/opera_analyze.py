"""Analysis of the OPeRA counterfactual replay: same context, same result list, human vs agent
choice. Every statistic is computed within the shown list, so the search engine is controlled.

Features of a (step, chosen item):
  bgpt_pct      within-list percentile of BehaviorGPT score, conditioned on the user's history
  bgpt_pct_nh   same without history
  bgpt_gain     bgpt_pct - bgpt_pct_nh (how much the user's own history explains the choice)
  bgpt_logit    s/tau - logsumexp(s_list/tau) + log n, tau fitted on human choices (2-fold by user)
Baselines (no BehaviorGPT): position, sponsored, price percentile, rating, query-title overlap pct.

Detectors: single features (AUROC, agent positive), logistic regression with user-grouped CV
(baselines vs baselines + BGPT), and user-level aggregation. CIs: bootstrap over users.
Usage: uv run python src/opera_analyze.py
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
TOK = re.compile(r"[a-z0-9]+")
BASE = ["pos_log", "sponsored", "price_pct", "rating", "overlap_pct"]
BG = ["bgpt_pct", "bgpt_pct_nh", "bgpt_gain", "bgpt_logit"]


def pct(vals, v):
    vals = np.asarray(vals, float)
    return float(((vals < v).sum() + 0.5 * (vals == v).sum() - 0.5) / max(len(vals) - 1, 1))


def fit_tau(rows):
    def nll(lt):
        t = np.exp(lt)
        return -sum(s[c] / t - logsumexp(np.array(list(s.values())) / t) for s, c in rows)
    return float(np.exp(minimize_scalar(nll, bounds=(-8, 2), method="bounded").x))


def feats(step, sc, scn, item, tau):
    ids = [it["id"] for it in step["shown"]]
    it = step["shown"][ids.index(item)]
    s = np.array(list(sc.values()))
    q = set(TOK.findall(step["query"]))
    ov = [len(q & set(TOK.findall(x["name"].lower()))) / max(len(q), 1) for x in step["shown"]]
    prices = [x["price"] for x in step["shown"] if x["price"]]
    f = {"pos_log": np.log1p(ids.index(item)), "sponsored": float(it.get("sponsored", False)),
         "price_pct": pct(prices, it["price"]) if it["price"] and len(prices) > 1 else 0.5,
         "rating": it.get("rating") or np.nan, "overlap_pct": pct(ov, ov[ids.index(item)]),
         "bgpt_pct": pct(s, sc[item]), "bgpt_pct_nh": pct(list(scn.values()), scn[item]),
         "bgpt_logit": sc[item] / tau - logsumexp(s / tau) + np.log(len(s))}
    f["bgpt_gain"] = f["bgpt_pct"] - f["bgpt_pct_nh"]
    return f


def boot_auc(y, x, g, B=1000, seed=0):
    y, x, g = map(np.asarray, (y, x, g))
    ok = ~np.isnan(x.astype(float))
    y, x, g = y[ok], x[ok].astype(float), g[ok]
    a = roc_auc_score(y, x)
    us = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in us}
    rng, bs = np.random.default_rng(seed), []
    for _ in range(B):
        i = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], x[i]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cv_auc(X, y, g):
    X = np.nan_to_num(np.asarray(X, float), nan=np.nanmedian(np.asarray(X, float)))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main():
    steps = [json.loads(l) for l in open(P / "opera_steps.jsonl")]
    S = {o["step_idx"]: o for o in map(json.loads, open(P / "opera_scores.jsonl"))}
    users = sorted({s["user"] for s in steps})
    fold = {u: i % 2 for i, u in enumerate(users)}
    tau = {}
    for k in (0, 1):  # tau for fold k is fitted on human choices of the other fold
        rows = [(S[i]["scores"], s["chosen"]) for i, s in enumerate(steps) if fold[s["user"]] != k]
        tau[k] = fit_tau(rows)
    print("tau by fold", tau)

    report = {"tau": tau, "n_steps": len(steps), "n_users": len(users), "conditions": {}}
    H = [feats(s, S[i]["scores"], S[i]["scores_nohist"], s["chosen"], tau[fold[s["user"]]]) for i, s in enumerate(steps)]
    print(f"history helps human choices: bgpt_pct mean {np.mean([h['bgpt_pct'] for h in H]):.3f} vs "
          f"no-history {np.mean([h['bgpt_pct_nh'] for h in H]):.3f}")
    for f in sorted(glob.glob(str(P / "opera_agent_*.jsonl"))):
        cond = Path(f).stem.replace("opera_agent_", "")
        A = {o["step_idx"]: o for o in map(json.loads, open(f))}
        idx = [i for i in range(len(steps)) if A.get(i, {}).get("choice")]
        if len(idx) < 50:
            print(cond, "skipped, valid", len(idx)); continue
        Af = {i: feats(steps[i], S[i]["scores"], S[i]["scores_nohist"], A[i]["choice"], tau[fold[steps[i]["user"]]])
              for i in idx}
        agree = np.mean([A[i]["choice"] == steps[i]["chosen"] for i in idx])
        r = {"n": len(idx), "agree_with_human": float(agree), "features": {}}
        print(f"\n=== {cond}: n={len(idx)}  agent picks same item as human: {agree:.2f}")
        print(f"{'feature':14s} {'human':>7s} {'agent':>7s}  paired-p   step AUROC [95% CI by user]   user-level AUROC")
        y = np.r_[np.zeros(len(idx)), np.ones(len(idx))]
        g = np.r_[[steps[i]["user"] for i in idx]] if False else np.array([steps[i]["user"] for i in idx] * 2)
        for k in BG + BASE:
            h, a = np.array([H[i][k] for i in idx], float), np.array([Af[i][k] for i in idx], float)
            ok = ~(np.isnan(h) | np.isnan(a))
            p = wilcoxon(h[ok], a[ok]).pvalue if np.any(h[ok] != a[ok]) else 1.0
            auc, lo, hi = boot_auc(y, np.r_[h, a], g)
            us = sorted({steps[i]["user"] for i in idx})
            uh = [np.nanmean([H[i][k] for i in idx if steps[i]["user"] == u]) for u in us]
            ua = [np.nanmean([Af[i][k] for i in idx if steps[i]["user"] == u]) for u in us]
            ok2 = ~(np.isnan(uh) | np.isnan(ua))
            uauc = roc_auc_score(np.r_[np.zeros(ok2.sum()), np.ones(ok2.sum())], np.r_[np.array(uh)[ok2], np.array(ua)[ok2]])
            print(f"{k:14s} {np.nanmean(h):7.3f} {np.nanmean(a):7.3f}  {p:8.1e}   {auc:.3f} [{lo:.3f},{hi:.3f}]        {uauc:.3f}")
            r["features"][k] = {"human_mean": float(np.nanmean(h)), "agent_mean": float(np.nanmean(a)), "paired_p": float(p),
                                "step_auroc": [auc, lo, hi], "user_auroc": float(uauc)}
        for name, cols in [("LR baselines", BASE), ("LR BGPT only", BG), ("LR baselines+BGPT", BASE + BG)]:
            X = [[H[i][k] for k in cols] for i in idx] + [[Af[i][k] for k in cols] for i in idx]
            pr = cv_auc(X, y, g)
            auc, lo, hi = boot_auc(y, pr, g)
            print(f"{name:20s} step AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  (5-fold CV grouped by user)")
            r["features"][name] = {"step_auroc": [auc, lo, hi]}
        report["conditions"][cond] = r
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(report, open(ROOT / "results/opera_replay.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
