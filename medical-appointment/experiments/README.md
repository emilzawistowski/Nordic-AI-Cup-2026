# Experiment Log: Medical Appointment Optimization

Track all experiment runs, hypotheses, parameters, metrics, protocol gates, and promotion decisions.

| Run ID | Git SHA | Hypothesis | CV Fold | Score | Acc | mIoU | Cond mIoU | P50 (s) | P95 (s) | Max (s) | Protocol Gate | Decision |
|--------|---------|------------|---------|-------|-----|------|-----------|---------|---------|---------|---------------|----------|
| exp1_baseline | 4fba6e49 | Unchanged baseline score | All 5 | 0.6346 | 0.9744 | 0.4081 | 0.4210 | 21.87 | 31.96 | 40.89 | PASS | Champion (Baseline) |
| exp2_evidence_alignment | 4fba6e49 | Guarded numeric & negation region alignment | All 5 | 0.6370 | 0.9744 | 0.4122 | 0.4253 | 17.82 | 26.21 | 40.49 | PASS | Reject (+0.0024 < 0.015 gate) |
| exp3_single_batch_10 | 4fba6e49 | Structured 10-question single generation batch | All 5 | 0.6394 | 0.9744 | 0.4161 | 0.4293 | 10.17 | 15.99 | 17.18 | PASS | PROMOTE (QA time -53% > 25% gate, score +0.0048) |
| exp4_fact_verifier | 4fba6e49 | Local fact verifier pre-filtering | All 5 | 0.6394 | 0.9744 | 0.4161 | 0.4293 | 11.03 | 17.13 | 17.96 | PASS | Reject (Score identical, verifier gate requires score +0.01) |




