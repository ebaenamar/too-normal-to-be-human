# Pilot: BehaviorGPT surprisal for human vs LLM-agent shopping sessions (2026-11-08)

## Gate 1: API mechanics (results/smoke.json), all PASS
- Deterministic: identical ids and scores across repeated calls.
- `filters: {"id": [...]}` scores any catalog item, consistent with the unfiltered ranking.
- History changes the ranking (top-20 overlap 0 vs empty history and vs `personalized: false`).
- Timing-free: 1 ms vs 20 min event spacing within the same hour gives identical scores.
- Throughput: about 1,300 calls/min with 8 workers, no 429s.
- Scores are raw similarities in a narrow band (about 1.30–1.49), so a temperature softmax (tau ≈ 0.022) is needed.

## Data
- Humans: WebShop MTurk demos on the full 1.18M-product store. 150 test sessions plus 300 calibration sessions, all with distinct instructions.
- Agents: claude-haiku-5.5 and deepseek-v4-flash via OpenRouter, 150 each, on the same instructions as the 150 test humans. Store is SQLite FTS5 BM25 over the same 1.18M products. Cost under $1.
- AgentTraj-L was discarded: it ran on the 1,000-product store, and none of its instructions overlap with the human demos (catalog confound).
- BehaviorGPT catalog: 62,231 items (everything touched or shown, plus a seeded uniform fill). Vendor scoring method from behaviorgpt-anomaly@vendor/ANOMALY_COMMIT.

## Results (AUROC, agent = positive; 95% bootstrap CI)
| Statistic | All agents | Notes |
|---|---|---|
| BGPT mean log-rank of the chosen item | 0.433 [0.377, 0.488] | agents less surprising |
| BGPT session z (excess surprisal) | 0.450 [0.394, 0.506] | |
| Paired by instruction, mean surprisal | agents lower, p = 0.001 | same sign for both models |
| **Engine-controlled: preference within the shown list** | **0.497 [0.441, 0.553]** | paired p = 0.59 |
| B1 click position | 0.592 | confounded by the different engine |
| B2 number of searches | 0.602 | |

## Reading
1. The "too normal" signal is real at population level but explained by the search engine: the pilot engine's result lists were already more model-aligned (63% vs 56% of the top 10 inside the model's top 100). Within the result set people actually saw, agents and humans choose equally "predictably".
2. WebShop gives a median of one scored step per session. A sequence model has almost no history to use, and MTurk humans with a fixed instruction behave like agents by design.
3. No statistic separates the classes usefully (best |AUROC − 0.5| ≈ 0.10, from cheap features).

## Implication for the paper
WebShop-style instructed tasks are the wrong testbed for a behavioral FM. A publishable path needs (a) the same engine for both classes, and (b) long, natural, multi-intent human sessions (e.g. ShopCART/OPeRA) against agents given comparable open-ended goals, where history matters. Alternatively, report this as a negative and methodological result: the engine confound inflates apparent separability.
