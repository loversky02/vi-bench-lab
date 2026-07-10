# Live baseline — cross-model agreement (ground-truth-free difficulty probe)

36 stratified MC rows, 5 frontier models via a self-hosted 9router gateway. `consensus` = mean top-vote fraction across models (**high = easy/agreed, low = genuinely hard/ambiguous**); `agreement` = each model vs the majority silver key. No ground truth is used — submit to V-Bench for official scores.

Coverage (answered/total): gpt-5.4-mini 0.972, claude-haiku-4-5-20251001 1.0, claude-sonnet-4-6 1.0, gpt-5.5 1.0, claude-opus-4-8 1.0

| Domain | n | consensus | gpt-5.4-mini | claude-haiku-4-5-20251001 | claude-sonnet-4-6 | gpt-5.5 | claude-opus-4-8 |
|---|---|---|---|---|---|---|---|
| culture | 6 | 0.7 | 0.833 | 0.833 | 0.5 | 0.5 | 0.833 |
| dialect | 6 | 0.9 | 1.0 | 0.833 | 1.0 | 0.833 | 0.833 |
| literature | 6 | 0.533 | 0.333 | 0.5 | 0.833 | 0.5 | 0.5 |
| logics | 6 | 0.6 | 0.667 | 0.833 | 0.667 | 0.167 | 0.667 |
| mathematics | 6 | 0.617 | 0.667 | 0.167 | 0.333 | 1.0 | 0.833 |
| physics | 6 | 0.7 | 0.833 | 0.5 | 0.667 | 0.833 | 0.667 |
| **overall** | 36 | 0.675 | 0.722 | 0.611 | 0.667 | 0.639 | 0.722 |
