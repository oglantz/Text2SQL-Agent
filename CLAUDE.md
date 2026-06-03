# Project: text-to-SQL agent distillation
- The agent loop, tools, and eval comparator are FROZEN after Phase 0. Never modify their behavior.
- The agent talks to an OpenAI-compatible endpoint; teacher and student differ only by base_url + model name.
- "Correct" = predicted SQL's result set matches gold result set (see src/eval.py). Guard against degenerate matches.
- Style: small, testable functions with explicit input/output contracts.