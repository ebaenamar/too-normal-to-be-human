"""Publication figures for the paper (vector PDF + 300-dpi PNG) in paper/figures/.

fig_auroc      AUROC with 95% user-cluster CIs: WebShop pilot (naive vs engine-controlled) and
               OPeRA replay (BGPT only, list baselines, both) for each agent condition.
fig_shift      ECDF of the within-list BehaviorGPT percentile of human vs agent choices.
fig_tells      Cheap behavioural tells: sponsored-click rate, list position, rating, query overlap.
fig_roc        ROC curves of the three detectors per agent condition.
Also writes paper/numbers.json with every number quoted in the text.
Usage: uv run python src/figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import opera_analyze as OA  # noqa: E402

OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
P = ROOT / "data/processed"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HUMAN = "#52514e"
CONDS = [("claude-haiku-5-5", "Haiku 5.5 + persona", "#2a78d6", "o"),
         ("claude-haiku-5-5_np", "Haiku 5.5, no persona", "#eb6834", "s"),
         ("deepseek-v4-flash", "DeepSeek V4 Flash + persona", "#1baf7a", "D")]
W = 5.5  # NeurIPS text width (in)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "lines.solid_capstyle": "round"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def user_boot(vals, users, B=2000, seed=0, stat=np.nanmean):
    vals, users = np.asarray(vals, float), np.asarray(users)
    us = np.unique(users)
    idx = {u: np.where(users == u)[0] for u in us}
    rng = np.random.default_rng(seed)
    bs = [stat(vals[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(B)]
    return float(stat(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load():
    steps = [json.loads(l) for l in open(P / "opera_steps.jsonl")]
    S = {o["step_idx"]: o for o in map(json.loads, open(P / "opera_scores.jsonl"))}
    users = sorted({s["user"] for s in steps})
    fold = {u: i % 2 for i, u in enumerate(users)}
    tau = {k: OA.fit_tau([(S[i]["scores"], s["chosen"]) for i, s in enumerate(steps) if fold[s["user"]] != k])
           for k in (0, 1)}
    F = lambda i, item: OA.feats(steps[i], S[i]["scores"], S[i]["scores_nohist"], item, tau[fold[steps[i]["user"]]])
    H = [F(i, s["chosen"]) for i, s in enumerate(steps)]
    A = {}
    for c, *_ in CONDS:
        rows = {o["step_idx"]: o for o in map(json.loads, open(P / f"opera_agent_{c}.jsonl"))}
        idx = [i for i in range(len(steps)) if rows.get(i, {}).get("choice")]
        A[c] = {"idx": idx, "f": {i: F(i, rows[i]["choice"]) for i in idx},
                "agree": float(np.mean([rows[i]["choice"] == steps[i]["chosen"] for i in idx]))}
    return steps, H, A, tau


def detectors(steps, H, A, c):
    idx = A[c]["idx"]
    y = np.r_[np.zeros(len(idx)), np.ones(len(idx))]
    g = np.array([steps[i]["user"] for i in idx] * 2)
    out = {}
    for name, cols in [("BGPT only", OA.BG), ("List baselines", OA.BASE), ("Baselines + BGPT", OA.BASE + OA.BG)]:
        X = [[H[i][k] for k in cols] for i in idx] + [[A[c]["f"][i][k] for k in cols] for i in idx]
        p = OA.cv_auc(X, y, g)
        out[name] = (y, p, g, OA.boot_auc(y, p, g))
    return out


def pilot_aurocs():
    """WebShop pilot: naive surprisal (sign flipped to 'predictability') vs engine-controlled."""
    a = json.load(open(ROOT / "results/pilot_auroc.json"))["auroc"]["all_agents"]["bgpt_mean_logrank"]
    naive = (1 - a["auroc"], 1 - a["ci"][1], 1 - a["ci"][0])
    rows = json.load(open(ROOT / "results/pilot_choiceset.json"))
    y = np.array([r["agent"] for r in rows]); x = np.array([r["c"] for r in rows], float)
    ctrl = OA.boot_auc(y, x, np.array([r["sid"] for r in rows]))  # sessions are independent units
    return naive, ctrl, int((y == 0).sum()), int((y == 1).sum())


def fig_auroc(det, pilot):
    naive, ctrl, nh, na = pilot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.08},
                                 sharex=True)
    for ax in (a1, a2):
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.38, 0.88)
    # (a) pilot
    labels = ["Surprisal of chosen item\n(engine differs)", "Within-list preference\n(engine controlled)"]
    for y, (m, lo, hi) in zip([1, 0], [naive, ctrl]):
        a1.plot([lo, hi], [y, y], color=INK2, lw=1.6, zorder=2)
        a1.plot(m, y, "o", ms=6, mfc=INK2, mec="white", mew=1.0, zorder=3)
        a1.annotate(f"{m:.2f}", (hi, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    a1.set_yticks([1, 0], labels); a1.set_ylim(-0.6, 1.6)
    a1.set_title("(a) WebShop pilot")
    a1.text(0.97, 0.97, f"{nh} humans vs {na} agents", va="top", ha="right", transform=a1.transAxes, fontsize=6.5, color=MUTED)
    a1.set_xlabel("AUROC (agent = positive)")
    # (b) replay
    dets = ["BGPT only", "List baselines", "Baselines + BGPT"]
    off = {c: o for (c, *_), o in zip(CONDS, [0.22, 0, -0.22])}
    for c, lab, col, mk in CONDS:
        for j, d in enumerate(dets):
            m, lo, hi = det[c][d][3]
            y = 2 - j + off[c]
            a2.plot([lo, hi], [y, y], color=col, lw=1.6, zorder=2)
            a2.plot(m, y, mk, ms=5, mfc=col, mec="white", mew=0.9, zorder=3)
            a2.annotate(f"{m:.2f}", (hi, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.3,
                        color=INK2)
    a2.set_yticks([2, 1, 0], dets); a2.yaxis.tick_right(); a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False); a2.set_ylim(-0.55, 2.55)
    a2.set_title("(b) OPeRA counterfactual replay (engine and context fixed)")
    a2.set_xlabel("AUROC (agent = positive), 5-fold CV grouped by user")
    h = [mpl.lines.Line2D([], [], color=col, marker=mk, lw=1.4, ms=4.5, mec="white", label=lab)
         for _, lab, col, mk in CONDS]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.2), columnspacing=1.6)
    save(fig, "fig_auroc")


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def ecdf(x):
    x = np.sort(np.asarray(x, float)); return np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1]


def fig_shift(steps, H, A):
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
    out = {}
    for ax, (c, lab, col, mk) in zip(axes, CONDS):
        idx = A[c]["idx"]
        h = np.array([H[i]["bgpt_pct"] for i in idx]); a = np.array([A[c]["f"][i]["bgpt_pct"] for i in idx])
        p = wilcoxon(h, a).pvalue
        xs, ys = ecdf(h); ax.step(xs, ys, where="post", color=HUMAN, lw=1.6)
        xs, ys = ecdf(a); ax.step(xs, ys, where="post", color=col, lw=1.6)
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
        ax.text(0.40, 0.62, "Human", color=INK, fontsize=7, ha="left")
        ax.text(0.80, 0.30, "Agent", color=INK, fontsize=7, ha="left")
        ax.annotate("", (0.80, 0.36), (0.74, 0.47), arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
        ax.text(0.03, 0.95, f"mean {h.mean():.2f} → {a.mean():.2f}\npaired $p$ = "+fmt_p(p),
                transform=ax.transAxes, va="top", fontsize=6.6, color=INK2)
        ax.set_title(lab, fontsize=7.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xticks([0, .5, 1])
        out[c] = {"human": float(h.mean()), "agent": float(a.mean()), "p": float(p), "n": len(idx)}
    axes[0].set_ylabel("Cumulative share of choices")
    fig.supxlabel("BehaviorGPT percentile of the chosen item within the shown list (1 = model's top pick)",
                  fontsize=8, y=-0.04)
    save(fig, "fig_shift")
    return out


def fig_tells(steps, H, A):
    tells = [("sponsored", "Sponsored result\nclicked (%)", 100, OA.np.nanmean),
             ("pos", "Position in list\n(mean rank)", 1, OA.np.nanmean),
             ("rating", "Star rating of\nchosen item", 1, OA.np.nanmean),
             ("overlap_pct", "Query–title word\noverlap (percentile)", 1, OA.np.nanmean)]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.75), sharey=True, gridspec_kw={"wspace": 0.15})
    rows = [("Human", HUMAN, "o", None)] + [(lab, col, mk, c) for c, lab, col, mk in CONDS]
    out = {}
    for ax, (k, title, scale, stat) in zip(axes, tells):
        for y, (lab, col, mk, c) in enumerate(reversed(rows)):
            if c is None:
                idx = range(len(steps)); f = lambda i: H[i]
            else:
                idx = A[c]["idx"]; f = (lambda cc: (lambda i: A[cc]["f"][i]))(c)
            v = [np.expm1(f(i)["pos_log"]) + 1 if k == "pos" else f(i)[k] for i in idx]
            m, lo, hi = user_boot(np.array(v, float) * scale, [steps[i]["user"] for i in idx])
            ax.plot([lo, hi], [y, y], color=col, lw=1.6)
            ax.plot(m, y, mk, ms=5, mfc=col, mec="white", mew=0.9)
            out.setdefault(k, {})[lab] = [m, lo, hi]
        ax.set_title(title, fontsize=7.2, fontweight="normal")
        ax.grid(axis="y", visible=False); ax.margins(x=0.18)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    save(fig, "fig_tells")
    return out


def fig_roc(det):
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (c, lab, col, mk) in zip(axes, CONDS):
        for d, color, ls in [("List baselines", INK2, "-"), ("Baselines + BGPT", col, "-"), ("BGPT only", col, (0, (3, 1.5)))]:
            y, p, g, (m, lo, hi) = det[c][d]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, lw=1.4 if ls == "-" else 1.2, ls=ls, label=f"{d} ({m:.2f})")
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=0)
        ax.set_title(lab, fontsize=7.8); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", frameon=False, fontsize=6.2, handlelength=1.8, borderaxespad=0.1)
    axes[0].set_ylabel("True positive rate")
    save(fig, "fig_roc")


def main():
    steps, H, A, tau = load()
    det = {c: detectors(steps, H, A, c) for c, *_ in CONDS}
    pilot = pilot_aurocs()
    fig_auroc(det, pilot)
    shift = fig_shift(steps, H, A)
    tells = fig_tells(steps, H, A)
    fig_roc(det)
    hist_gain = {"human_with": float(np.mean([h["bgpt_pct"] for h in H])),
                 "human_without": float(np.mean([h["bgpt_pct_nh"] for h in H]))}
    nums = {"tau": tau, "n_steps": len(steps), "n_users": len({s["user"] for s in steps}),
            "list_size_median": float(np.median([len(s["shown"]) for s in steps])),
            "hist_len_median": float(np.median([s["t"] for s in steps])),
            "pilot": {"naive": pilot[0], "controlled": pilot[1], "n_human": pilot[2], "n_agent": pilot[3]},
            "detectors": {c: {d: v[3] for d, v in det[c].items()} for c in det},
            "agree": {c: A[c]["agree"] for c in A}, "n_valid": {c: len(A[c]["idx"]) for c in A},
            "shift": shift, "tells": tells, "history": hist_gain}
    json.dump(nums, open(ROOT / "paper/numbers.json", "w"), indent=1, default=float)
    print(json.dumps(nums, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
