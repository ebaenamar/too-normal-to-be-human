# OPeRA counterfactual replay (2026-11-08)

## Design
- Humans: OPeRA full (CC-BY-4.0). 54 real users on live Amazon, 692 sessions. 384 clicks on a search-result list (45 users), with a median of 13 history events in the API call.
- At each human click, an LLM agent acting for that user picks from the **identical** list (median 55 items). The agent sees the user's self-reported persona, their last 12 events and the same query.
- Engine, query and context are therefore fixed by construction; only the chooser differs.
- Conditions: claude-haiku-5.5 with persona, claude-haiku-5.5 without persona, deepseek-v4-flash with persona. DeepSeek returned a valid choice in only 247/384 steps.
- BehaviorGPT: one `/vector/search` call per step (query, real user history, `filters.id` = the shown list) gives a score for every shown item. A second call with empty history is the control.
- Catalog: 74,347 items (WebShop pilot + 12,116 OPeRA products, title and price only).
- Features are within-list, so they are engine-controlled. tau is fitted on human choices, 2-fold by user. Step-level AUROC CIs come from a user-cluster bootstrap. The LR uses 5-fold CV grouped by user.

## Results (agent = positive)
| | haiku + persona | haiku, no persona | deepseek + persona |
|---|---|---|---|
| Agent picks the same item as the human | 10% | 14% | 19% |
| BGPT within-list percentile, human vs agent | 0.59 vs 0.64 (p = 0.014) | 0.59 vs 0.67 (p = 3e-5) | 0.58 vs 0.67 (p = 3e-4) |
| BGPT percentile, step AUROC | 0.54 [0.51, 0.58] | 0.58 [0.54, 0.61] | 0.58 [0.52, 0.63] |
| BGPT percentile, user-level AUROC | 0.61 | 0.65 | 0.62 |
| Best single baseline (step AUROC) | position 0.72 | position 0.68 | rating 0.62 |
| LR, baselines only | 0.80 | 0.74 | 0.62 |
| LR, BGPT only | 0.54 | 0.57 | 0.57 |
| LR, baselines + BGPT | 0.80 | 0.75 | 0.64 |

Baseline features: list position, sponsored, price percentile, star rating, and query–title word overlap.

## Reading
1. **H1 "too normal" holds once the engine is controlled.** In all three conditions agents choose items that BehaviorGPT ranks higher than the human's choice in the same context (paired p ≤ 0.014). This is the opposite of the vendor's anomaly assumption ("bots look surprising").
2. **The effect is small and redundant with cheap signals.** Step AUROC is 0.54–0.58. Adding BGPT to a logistic regression over five list features improves it by at most 0.015, within the CI.
3. **The real tells are cheap.** Agents skip sponsored results (34% of human clicks vs 5% for haiku), go deeper in the list, and favour higher ratings and more literal query matches.
4. **Persona does not hide the agent.** Removing the persona slightly increases BGPT separability but decreases baseline separability. DeepSeek is the hardest agent to detect (LR 0.62–0.64).
5. **History helps only a little.** Conditioning on the user's history raises the percentile of human choices by just 0.02, and raises the agents' by more (0.03–0.05), because agents see the recent history in their prompt and stay consistent with it.

## Caveats
- One choice per step. Agents do not generate whole sessions, so session-level behaviour (queries, dwell, scroll) is not tested.
- OPeRA items have only title and price in the catalog, with no categories or images.
- 37% of DeepSeek answers were invalid or empty, which may select its easier steps.
- 45 users is a small sample at user level.
