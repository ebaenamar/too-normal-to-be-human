# Too Normal to Be Human

**Can a behavioral foundation model tell LLM agents from the humans they act for?**

We tested BehaviorGPT, Unbox AI's 12.5B-parameter model trained on 150B e-commerce actions, against real LLM shopping agents. The agents are not anomalous. They are *too normal*: the model ranks their choices higher than the human's. The signal is weak, though, and five cheap features of the result list detect agents far better.

![Detection performance](paper/figures/fig_auroc.png)

## Why it matters

LLM agents now search, compare and buy inside people's accounts, with their history, preferences and payment methods. Platforms need to know when a choice was made by the user or by an agent acting for them. This matters for fraud, account takeover and, in sensitive settings such as government or defense networks, insider-risk monitoring.

Behavioral foundation models look like the natural tool: they assign a likelihood to every next action given a user's history. The vendor's recommended detector is anomaly scoring (surprisal plus CUSUM), which assumes non-human behavior is surprising. We test that assumption.

## What we did

### 1. Pilot on WebShop: we found a confound

- **Data:** 150 human sessions from WebShop, and 300 agent sessions (Claude Haiku 5.5 and DeepSeek V4 Flash) on the same instructions over the same 1.18M products.
- **Naive result:** with the vendor's method, agents look less surprising (AUROC 0.57, paired p ≈ 0.001).
- **Why it is spurious:** the agents' search engine returned lists that were already more aligned with the model. 63% of their top-10 results fall in the model's top 100, against 56% for the humans' engine.
- **Controlled result:** scoring each choice within the list that chooser actually saw, AUROC is 0.50 and the effect disappears.

### 2. Counterfactual replay on OPeRA: our protocol

OPeRA logs real users shopping on live Amazon, together with their self-reported personas. We use 384 clicks on result lists from 45 users, with a median of 55 results per list and 13 history events per call.

At every human click, an LLM agent acting for that same user chooses from the **identical list with the identical history** and the user's persona. Engine, query and context are therefore fixed by construction; only the chooser differs. This is also the realistic threat model: a delegated agent operating inside a legitimate account.

BehaviorGPT scores every item in the list with one API call. We then compare the human's choice with the agent's using three statistics:

- the item's percentile within the list;
- a conditional logit;
- the history gain (how much the user's history raises the item's percentile).

![Agents pick what the model expects](paper/figures/fig_shift.png)

## Results

| | Haiku 5.5 + persona | Haiku 5.5, no persona | DeepSeek V4 Flash + persona |
|---|---|---|---|
| Agent picks the same item as the human | 10% | 14% | 19% |
| BehaviorGPT percentile, human → agent | 0.59 → 0.64 (p = 0.014) | 0.59 → 0.67 (p = 3e-5) | 0.58 → 0.66 (p = 3e-4) |
| BehaviorGPT only, step AUROC | 0.54 | 0.57 | 0.57 |
| BehaviorGPT, user-level AUROC | 0.61 | 0.65 | 0.62 |
| Five list features (logistic regression) | **0.80** | **0.74** | **0.62** |
| List features + BehaviorGPT | 0.80 | 0.75 | 0.64 |

The five list features are list position, sponsored flag, price percentile, star rating, and query–title word overlap. AUROCs are cross-validated, grouped by user, with 95% CIs from a user bootstrap; see the paper.

**Findings**

1. **Agents are too normal, not anomalous.** In all three conditions they choose items the model ranks higher than the human's choice. An anomaly detector pointed at agents is aimed the wrong way.
2. **The signal is weak and redundant.** BehaviorGPT adds at most 0.015 AUROC over five features any platform already logs.
3. **The real tells are cheap.** Humans click sponsored results in 34% of steps, Haiku in 5%. Agents prefer higher-rated items and more literal query matches, and Haiku goes deeper into the list.
4. **Persona does not hide the agent.** The model family matters more: DeepSeek is the hardest agent to detect.
5. **Methodological lesson.** Comparing humans and agents across search engines produces convincing but spurious effects. Score choices relative to what each chooser was shown.

![Cheap tells](paper/figures/fig_tells.png)

## Paper

NeurIPS-format draft: [`paper/main.pdf`](paper/main.pdf), with LaTeX sources in `paper/`. Every number in the text is regenerated into `paper/numbers.json` by `src/figures.py`.

## Repository

```
src/
  smoke.py           API mechanics checks (determinism, id filter, history effect, timing, throughput)
  parse.py           WebShop human/agent sessions -> unified format
  build_items.py     WebShop 1.18M products -> jsonl + SQLite FTS5 index
  agent_env.py       text WebShop environment + OpenRouter agent (pilot)
  build_catalog.py   build and upload the BehaviorGPT catalog
  score.py           vendor-method scoring (head top-100 + id-filtered tail)
  analyze.py         pilot AUROCs vs baselines
  choice_set.py      engine-controlled within-list statistic (pilot)
  opera_parse.py     OPeRA -> per-user event streams + replayable click steps
  opera_agent.py     counterfactual replay agent (persona, history, identical list)
  opera_catalog.py   add OPeRA products to the catalog
  opera_score.py     one BehaviorGPT call per step, with and without history
  opera_analyze.py   paired tests, AUROCs, grouped-CV detectors
  figures.py         publication figures + paper/numbers.json
results/             PILOT.md, OPERA_REPLAY.md and JSON outputs
paper/               LaTeX sources, figures (PDF + PNG), compiled PDF
```

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), a BehaviorGPT API key and an OpenRouter key.

1. Set up the environment and the vendored anomaly toolkit, pinned to the commit we used:

   ```bash
   uv sync
   git clone https://github.com/Unbox-AI/behaviorgpt-anomaly vendor/behaviorgpt-anomaly
   git -C vendor/behaviorgpt-anomaly checkout $(cat vendor/ANOMALY_COMMIT)
   ```

2. Create a `.env` file (never commit it) with:

   ```
   UNBOXAI_API_KEY=...
   OPENROUTER_API_KEY=...
   ```

3. Download OPeRA (CC-BY-4.0) into `data/raw/opera/` from https://huggingface.co/datasets/NEU-HAI/OPeRA. Then run the replay pipeline:

   ```bash
   uv run python src/opera_parse.py
   uv run python src/opera_catalog.py
   uv run python src/opera_agent.py --model anthropic/claude-haiku-5.5
   uv run python src/opera_agent.py --model anthropic/claude-haiku-5.5 --no-persona
   uv run python src/opera_agent.py --model deepseek/deepseek-v4-flash
   uv run python src/opera_score.py
   uv run python src/opera_analyze.py
   uv run python src/figures.py
   ```

Cost: under US$10 in LLM calls; the BehaviorGPT API was free during early access.

## Limitations

- We judge single decisions, not full agent sessions. Query writing, navigation and timing are out of scope, and BehaviorGPT ignores time below the hour anyway.
- OPeRA products enter the catalog with title and price only.
- 36% of DeepSeek's answers were invalid.
- 45 users is a small sample for user-level claims.

## Data and ethics

- **Public data only:** WebShop and OPeRA (CC-BY-4.0).
- **No sensitive data:** BehaviorGPT's terms allow the vendor to train on uploaded content, so the hosted model is unsuitable for sensitive (e.g. defense) data regardless of accuracy.
- **Dual use:** detecting delegated agents can protect users, but crude heuristics such as ad avoidance would also block legitimate assistive agents.
