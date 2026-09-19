# Attempt History — medical-appointment

## Scoring formula
Score = 0.5 × Accuracy + 0.5 × Mean_tIoU
Accuracy ceiling ~0.974 already reached. Remaining gain is in tIoU.

## Baselines
| # | Date | Score | Accuracy | tIoU | What changed |
|---|------|-------|----------|------|--------------|
| 0 | 2026-09-17 | ~0 | - | - | Baseline (rule-based edge detection) |
| 1 | 2026-09-18 | 0.xxx | 0.969 | ~0.3 | Added MLX Whisper ASR + Qwen3-4B QA |
| 2 | 2026-09-19 | 0.6394 | 0.974 | 0.416 | Batch QA (all 10 questions at once), guarded evidence |

## Dead ends
- exp2: Negation-aligned evidence windows → +0.0024 tIoU, below threshold
- exp4: fact_verifier hard-negative pre-filter → no improvement over Qwen3 batching

## Next ideas (ranked by expected impact)
1. Second LLM call to re-quote exact shortest phrase (shorter spans → higher tIoU)
2. Bigger model: Qwen3-8B or 14B (better evidence selection)
3. Whisper word-confidence filtering (remove low-confidence words from span)
4. Qwen3 thinking mode for hard near-miss questions

---
*Append new attempts below this line*
