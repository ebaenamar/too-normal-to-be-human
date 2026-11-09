"""Figure for the official-WebShop trajectory experiment (same style as figures.py).

fig_traj  (a) AUROC of trajectory / BehaviorGPT / combined detectors, incl. the
              purchase-feature-free control; (b) behavioural tells, human vs agents.
Usage: uv run python src/figures_traj.py -> paper/figures/fig_traj.{pdf,png}, paper/numbers_traj.json
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from figures import GRID, HUMAN, INK, INK2, MUTED, W, save, user_boot  # noqa: E402  (also sets rcParams)

AG = [("claude-haiku-5-5", "Haiku 5.5", "#2a78d6", "o"), ("deepseek-v4-flash", "DeepSeek V4 Flash", "#1baf7a", "D")]


def main():
    rep = json.load(open(ROOT / "results/ws_trajectory.json"))["conditions"]
    rob = json.load(open(ROOT / "results/ws_trajectory_robust.json"))
    T = [json.loads(l) for l in open(ROOT / "data/processed/ws_session_table.jsonl")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})

    dets = [("Trajectory", lambda c: rep[c]["detectors"]["Trajectory features"]),
            ("Trajectory, no\npurchase features", lambda c: rob[f"{c}|all sessions|traj (no purchase feats)"][:3]),
            ("BehaviorGPT only", lambda c: rep[c]["detectors"]["BGPT only"]),
            ("Trajectory +\nBehaviorGPT", lambda c: rep[c]["detectors"]["Trajectory + BGPT"])]
    for j, (lab, get) in enumerate(dets):
        for k, (c, name, col, mk) in enumerate(AG):
            m, lo, hi = get(c)
            y = len(dets) - 1 - j + (0.15 if k == 0 else -0.15)
            a1.plot([lo, hi], [y, y], color=col, lw=1.6)
            a1.plot(m, y, mk, ms=5, mfc=col, mec="white", mew=0.9)
            a1.annotate(f"{m:.2f}", (hi, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.3, color=INK2)
    a1.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    a1.set_yticks(range(len(dets))[::-1], [d[0] for d in dets]); a1.grid(axis="y", visible=False)
    a1.set_xlim(0.45, 0.95); a1.set_ylim(-0.6, len(dets) - 0.4)
    a1.set_xlabel("AUROC (agent = positive)")
    a1.set_title("(a) Session-level detection")

    tells = [("n_item_views", "Items viewed"), ("n_back_to_results", "Returns to results"), ("n_queries", "Queries issued")]
    rows = [("Human", HUMAN, "o", "human")] + [(n, col, mk, c) for c, n, col, mk in AG]
    nums = {}
    for i, (k, lab) in enumerate(tells):
        for r_i, (name, col, mk, src) in enumerate(rows):
            sub = [t for t in T if t["source"] == src]
            m, lo, hi = user_boot([t[k] for t in sub], [t["pair"] for t in sub])
            y = len(tells) - 1 - i + (1 - r_i) * 0.24
            a2.plot([lo, hi], [y, y], color=col, lw=1.6)
            a2.plot(m, y, mk, ms=5, mfc=col, mec="white", mew=0.9)
            nums.setdefault(k, {})[name] = [m, lo, hi]
    a2.set_yticks(range(len(tells))[::-1], [t[1] for t in tells]); a2.grid(axis="y", visible=False)
    a2.set_ylim(-0.6, len(tells) - 0.4); a2.set_xlabel("Mean per session (95% CI)")
    a2.set_title("(b) What gives agents away")
    h = [mpl.lines.Line2D([], [], color=col, marker=mk, lw=1.4, ms=4.5, mec="white", label=n) for n, col, mk, _ in rows]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17), handlelength=1.6)
    save(fig, "fig_traj")
    json.dump({"tells": nums}, open(ROOT / "paper/numbers_traj.json", "w"), indent=1)


if __name__ == "__main__":
    main()
