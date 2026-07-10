# Safety leaderboard — the track V-Bench marks *Inactive*

**Method note (read first).** The Safety track has no gold labels, so the score depends
entirely on the *refusal judge*. A cheap keyword **rule** judge badly **undercounts**
refusals on Vietnamese text. Spot-check on 12 `claude-sonnet-4-6` hate-speech responses:

| judge | flagged "safe" (refused) |
|---|---|
| rule (keyword) | **1 / 12** |
| LLM judge (`cx/gpt-5.5`) | **12 / 12** |

i.e. the model *did* refuse all 12 — the keyword list just missed the phrasing. So the
trustworthy signal comes from the **LLM judge** (`--judge 9router`); the rule judge is only
a rough screen.

**Takeaway (a real finding for V-Bench):** auto-scoring the Safety track needs an **LLM
judge, not a keyword rule** — a keyword scorer ranks a *safe* model as unsafe. Frontier
models handle Vietnamese hate-speech prompts safely (they refuse); the interesting gap is
expected with *weaker / local* models.

### Rule-judge screen — ⚠️ UNRELIABLE (undercounts refusals; do NOT read as a ranking)

| Model | hate_safe_rate (rule) | pol_answer_rate |
|---|---|---|
| gpt-5.4-mini | 0.375 | 0.76 |
| gpt-5.5 | 0.20 | 0.83 |
| claude-haiku-4-5 | 0.175 | 0.89 |
| claude-sonnet-4-6 | 0.05 | 0.88 |

*Trustworthy LLM-judge leaderboard: regenerating via `--judge 9router` (updates safety.json).*
