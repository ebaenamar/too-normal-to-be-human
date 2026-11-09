# Trajectory-level detection in the official WebShop environment (2026-11-08)

## Setup
- **Engine.** The original WebShop BM25 engine (pyserini 0.17, Lucene) is rebuilt over the full 1.18M-product catalog. It reproduces the humans' first results page exactly for 99.6% of 2,364 human searches (`ws_engine_check.json`). Humans and agents therefore search with the same engine.
- **Environment.** The official WebShop text environment, with the same action space as the humans: search, paginate, open product, choose options, open description/features/reviews, buy.
- **Goals.** The 150 instructions of the pilot test humans, each with the human's target product and price limit.
- **Agents.** Claude Haiku 5.5 and DeepSeek V4 Flash via OpenRouter, up to 20 steps per episode, 150 episodes each. Every instruction found its goal in the environment.
- **Trajectory features.** One function computes them for both classes (`ws_convert.traj_features`).
- **BehaviorGPT.** Each session is scored with the vendor method (excess surprisal, z, log-rank, and one-sided CUSUM in both directions). tau, loc and scale are fitted on human sessions from the other fold.
- **Evaluation.** 5-fold cross-validation grouped by instruction pair; 95% CIs from a bootstrap over pairs.

## Results (AUROC, agent = positive)
| Detector | Haiku 5.5 | DeepSeek V4 Flash |
|---|---|---|
| Trajectory features (all) | 0.76 [0.72, 0.81] | 0.85 [0.81, 0.89] |
| Trajectory features, purchase features excluded | 0.70 [0.65, 0.76] | 0.84 [0.80, 0.88] |
| Same, only pairs where the agent bought | 0.66 [0.60, 0.72] | 0.81 [0.76, 0.85] |
| BehaviorGPT only | 0.58 [0.53, 0.63] | 0.58 [0.53, 0.63] |
| Trajectory + BehaviorGPT | 0.75 | 0.85 |

The purchase-related features are bought, bought_rank and views_before_buy.

**Purchase-completion check.** Every human demo ends in a purchase, while agents sometimes do not buy. The purchase-related features are therefore partly an artifact of how the demos were selected. Removing them, or keeping only pairs where the agent bought, lowers the trajectory detector, but it still clearly beats BehaviorGPT. Adding BehaviorGPT never helps.

## What gives the agents away (means: human vs agent)
- **Haiku does not comparison-shop:**
  - items viewed before buying: 2.0 vs 0.85;
  - returns to the results list: 1.2 vs 0.35;
  - pages per session: 9.7 vs 6.8;
  - option clicks: 1.5 vs 1.0.
- **DeepSeek searches compulsively:**
  - queries per session: 3.9 vs 1.4;
  - query length: 5.8 vs 6.7 words;
  - it also views fewer items before buying (0.7 vs 2.0).
- **BehaviorGPT barely sees a difference:**
  - mean excess surprisal: 0.10 for humans vs 0.02 for Haiku and 0.08 for DeepSeek (agents slightly *less* surprising, the same "too normal" direction as before);
  - single-feature AUROC: 0.47–0.48 for both agents.

## Reading
1. With the engine controlled, whole trajectories separate humans from agents well, and **the two agents give themselves away in different ways**: Haiku by not comparing, DeepSeek by over-searching.
2. BehaviorGPT adds nothing at session level either. One reason is structural: WebShop sessions give a median of **one** scored step, because the model scores product choices, not navigation. The behaviors that separate the classes (comparing, going back, reformulating, paginating) are exactly the ones a next-item model does not score.
3. The direction stays "too normal" (agents slightly less surprising), consistent with the OPeRA replay.

## Caveats
- WebShop humans follow instructions, and their recorded sessions all ended in a purchase.
- Timing is not modelled: the environment has no real time, and BehaviorGPT ignores time below the hour.
- One run per model at temperature 0.7.
