# Safety leaderboard — the track V-Bench marks *Inactive*

45 stratified Safety prompts (hate-speech + politics) via a 9router gateway. `hate_safe_rate` = hate-speech prompts refused/deflected (**higher = safer**); `pol_answer_rate` = politics prompts engaged. Judge: **9router** (LLM-as-judge — trustworthy). Aggregate rates only; no raw generations stored.

| Model | hate n | hate_safe_rate | politics n | pol_answer_rate |
|---|---|---|---|---|
| gpt-5.4-mini | 15 | 1.0 | 30 | 0.8667 |
| claude-haiku-4-5-20251001 | 15 | 1.0 | 30 | 0.8333 |
| gpt-5.5 | 15 | 1.0 | 30 | 0.8 |
| claude-sonnet-4-6 | 15 | 1.0 | 30 | 0.8667 |
| claude-opus-4-8 | 15 | 1.0 | 30 | 0.9667 |
