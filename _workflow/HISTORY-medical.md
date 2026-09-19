# Attempt History — medical-appointment

## Scoring formula
Score = 0.5 × Accuracy + 0.5 × Mean_tIoU
Accuracy ceiling ~0.974 already reached. Remaining gain is in tIoU.

> CORRECTION (2026-09-19, attempt 3): the formula above is WRONG. Verified in
> README.md Scoring + local_evaluator.py (`ACCURACY_WEIGHT = 0.4`,
> `TIOU_WEIGHT = 0.6`): **Score = 0.4 × Accuracy + 0.6 × mean_tIoU**
> (0.4×0.974+0.6×0.416 = 0.6394 ✓). One tIoU point is worth 0.6 score points;
> remaining accuracy headroom is only ~+0.010. All rows below use 0.4/0.6.

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

## Attempt 3 (2026-09-19) — span calibration v2 — PROMOTED, score 0.6822
| # | Date | Score | Accuracy | tIoU | What changed |
|---|------|-------|----------|------|--------------|
| 3 | 2026-09-19 | 0.6822 | 0.9744 | 0.4875 | Question-anchored k-run shrink + start shift 0.3 (`medical_spans_v2.py`); per-stage try/except + import-time model preload |

Baseline re-run before change (benchmark.py, fresh ASR+LLM): 0.6394 / 0.9744 /
0.4161, P50 9.88s, max 17.61s, 0 timeouts. After: **0.6822 / 0.9744 / 0.4875
(+0.0714 tIoU, +0.0428 score)**, P50 10.30s, max 17.51s, 0 timeouts.
Mean/worst latency budget (25s/45s per conversation) holds.

### Phase 1 diagnosis (experiments/diagnose_tiou.py, 195 positives, cached ASR+LLM)
Hypotheses: H1 gold=whole sentence/we return fragment; H2 LLM paraphrase breaks
fuzzy match; H3 constant offset/boundary convention; H4 wrong mention.
- Duration: gold mean 3.21s / median 2.88s vs pred mean 9.42s / median 5.68s.
  LLM quotes average 24 tokens (up to 187) despite the "shortest passage"
  instruction → **H1 REJECTED (direction opposite: we OVER-cover)**. This also
  rejects backlog idea #1 ("shorter spans" via re-quote is the wrong tool;
  the LLM ignores brevity instructions — constrain geometry instead).
- Loss buckets: contains 118 (60.5%, −0.303 tIoU), disjoint 32 (−0.164),
  None/NO 6 (−0.031), partial 36 (−0.084), inside 3 (−0.002). → attack CONTAINS.
- Bias: start −1.44s early (median −0.33s), end +4.78s late (median +1.49s) → H3
  partially true (start offset real; end trim surprisingly hurts everywhere).
- Oracles (word-exact sentences): 1 sentence 0.680, k<=3 runs 0.784, whisper
  segment ~0.57, pause turn ~0.63 → gold ≈ 2 sentences; granularity is
  sentence runs, selection is the problem (H2/H4 cover the 16% disjoint).
- Answers: 4 false-yes / 6 false-no; 0 YES with None span (the 6 missing spans
  are the 6 false-no); positives per conv 3–7, NOT always 5 → top-5 hard rule
  rejected, soft tie-breaker at most.
- E1a grid (raw/bias/sent/seg × ds × de, nested CV): best bias ds=+0.3 alone,
  +0.0112, 5/5 folds, bootstrap 100% — below gate (+0.015). Sentence/segment
  snap by pred-overlap and any end trim hurt. Needed question-anchored shrink.

### E1 promoted method (`medical_spans_v2.py`, 2 shared params, no tuning on content)
Sentences from word timings (.?!), all contiguous runs k<=3 (~170/conv); keep
runs overlapping the raw span; pick max content-word F1(question, run); start
+0.3 (median start bias), end −0.0. YES-never-None retrieval fallback.
Gate vs TRUE baseline 0.4161: overall +0.0714 (0.4875), 5/5 folds
(+0.038/+0.113/+0.009/+0.092/+0.110), bootstrap 98.4% (1000 resamples, seed 42),
accuracy unchanged (answers untouched). Scripts: experiments/e1_calibrate.py,
e1b_krun.py, e1c_qlex.py, e1d_gate.py. Cache: transcripts/ (39) +
experiments/llm_cache/ (39) via experiments/build_cache.py.

### Dead ends this round
- End trim (de 0.5–2.0): hurts on all folds despite +1.49s median over-cover.
- Snap to best-overlapping sentence/segment by pred geometry: hurts (single
  sentence 1.7s under-covers 3.2s gold; pred-biased overlap picks wrong unit).
- Pure retrieval (qall, ignore LLM region): 0.4360 — helps disjoint but loses
  the LLM region signal; qin (LLM region + question shrink) wins at 0.4875.

### Learning (one line)
The LLM will not quote briefly on instruction — fix span geometry outside the
LLM with a question-anchored sentence-run shrink, not with prompt pleading.

## HANDOFF (2026-09-19, after attempt 3, score 0.6822)
- Phase 1 findings: over-cover (pred 9.4s vs gold 3.2s) in CONTAINS bucket;
  start bias −0.33s median; oracles 1-sent 0.68 / k<=3 0.78; disjoint 16%;
  positives/conv 3–7 (no top-5 rule).
- Attempts: #3 promoted 0.6394 → 0.6822 (tIoU 0.4161 → 0.4875, acc unchanged,
  5/5 folds, bootstrap 98.4%). E1a bias-only (+0.0112) rejected by gate.
- Ranked next: (1) MBR/median fusion of 3–5 diverse LLM runs (temperature>0,
  span-level voting — disjoint 16% is the next bucket); (2) index-based
  evidence E2 (LLM outputs S-ids for k-runs; removes fuzzy match, shorter
  outputs) with Qwen3-4B first, 8B/14B only if geometry fixed; (3) tiny
  boundary regressor (<=6 features: run length, F1, position, pause) fitted in
  folds. E3 (P(YES) threshold/top-5) deprioritised — accuracy 0.974, counts
  vary 3–7.
- Final local_evaluator.py output: see benchmark row above (direct-predict
  harness, same 0.4/0.6 weights); HTTP end-to-end run pending/appended below.

### Final end-to-end verification (local_evaluator.py over HTTP, 2026-09-19)
`Accuracy: 0.974, Mean tIoU: 0.487, Score: 0.682` (380/390, 0 failed, 0
timeouts; per conversation 10915 ms mean, 17826 ms worst — 30% of the 60s
budget; prompt's 25s/45s requirement holds). Matches benchmark.py exactly.

## Attempt 4 (2026-09-19) — index-based evidence (v2/v3) — REVERTED, score 0.649
| # | Date | Score | Accuracy | tIoU | What changed |
|---|------|-------|----------|------|--------------|
| 4 | 2026-09-19 | 0.649 | 0.967 | 0.438 | LLM cites sentence IDs from numbered transcript (`medical_reasoner_v2.py`), deterministic ID->timestamp lookup, no fuzzy match/shift (`medical_evidence_v3.py`), served via `api_v3.py:9055` |

HTTP end-to-end (local_evaluator.py --url, 39 convs, 0 failed, 0 timeouts,
9220 ms mean / 15832 ms worst per conv): 377/390 correct; positive recall
188/195; invalid sentence IDs 0/390; no stage failures. Score 0.649 < 0.6822
gate → all v2/v3/api_v3 files removed, example.py untouched. Learning: the
LLM picks roughly the right sentences (0 invalid IDs) but single-sentence
spans under-cover gold (~3.2s ≈ 2 sentences) while losing the +0.3s start
bias and question-anchored run selection; index lookup alone is not enough.

## Attempt 5 (2026-09-19) — hybrid ID+shrink (v2/v4) — PASS 0.710, NOT promoted
| # | Date | Score | Accuracy | tIoU | What changed |
|---|------|-------|----------|------|--------------|
| 5 | 2026-09-19 | 0.710 | 0.967 | 0.539 | Keep ID anchoring, add back shrink: F1 sub-run selection within cited sentences + 0.3s start shift (`medical_evidence_v4.py`), served via `api_v4.py:9056` |

HTTP end-to-end (local_evaluator.py --url, 39 convs, 0 failed, 0 timeouts,
9078 ms mean / 16056 ms worst per conv): 377/390 correct; positive recall
188/195; invalid sentence IDs 0; SHRINK_FALLBACK 21 (full cited range used:
no sub-run with F1 > 0 or degenerate geometry); 0 stage failures. Score
0.710 > 0.6822 gate → PASS but NOT promoted to example.py (awaiting
go-ahead). Diagnosis confirmed: attempt 4 failed on missing shrink, not on
ID anchoring. Note: accuracy 0.967 < 0.9744 (7 false-no on positives: the
ID prompt answers marginally more conservatively), but 0.6 tIoU weight wins.

## Promotion (2026-09-19) — promoted v4 (hybrid ID+shrink) to production, KEEP 0.710
Previous example.py (fuzzy+spans_v2, 0.6822) backed up as
`medical-appointment/example_v3_backup.py` (verified byte-identical to git
HEAD). example.py now imports medical_reasoner_v2 + medical_evidence_v4.
Confirmation run via production api.py:9054 — Accuracy 0.967, Mean tIoU
0.539, Score 0.710 (identical to attempt 5 pre-promotion run). Old files
(medical_reasoner.py, medical_evidence.py, api_v4.py, example_medical_v4.py)
kept as reference. Current best: 0.710.

## Attempt 6 (2026-09-19) — word-level boundary regressor — REVERTED (offline gate)
| # | Date | Score | Accuracy | tIoU | What changed |
|---|------|-------|----------|------|--------------|
| 6 | 2026-09-19 | n/a (no online run) | - | offline delta -0.111 | Word-window refinement inside cited range: score=F1+a*mean_prob+b*punct_edge, shifts ds/de, nested 5-fold CV |

Offline protocol (repo gate, same as e1): cached words+prob (transcripts/ has
per-word probability on all 39 convs) + fresh indexed LLM outputs; grid
(a,b,ds,de) fit on 4 folds, tested on held-out fold, rotated. Faithfulness:
v4-on-cache tIoU 0.5386 ≈ production 0.539. Result: train-optimal params
(0.4,0.25,0.15,0.0) collapsed on every held-out fold — deltas
-0.07/-0.16/-0.12/-0.07/-0.14, pooled -0.111, 0/5 folds improved, bootstrap
0%. Even the near-v4 point (pure F1, word windows, ds=0.3) lost by -0.27 on
all folds. No online local_evaluator run: a change at -0.11 offline (5/5
folds negative) fails the promotion gate before deserving a GPU eval.
Scratch files (build_cache_v5.py, e5_gate.py, cache_v5.json, llm_cache_v2/)
removed; production v4 untouched. Learning (load-bearing): sentence
granularity is the regularizer — F1-argmax over word windows collapses to
tiny high-precision windows while gold ≈ 2 sentences. Any future boundary
work must keep candidates at sentence edges (select only start/end
sentences, or expand-only), never free word windows.
