"""Span calibration v2 (attempt 3, promoted through the gate).

Finding (Phase 1, experiments/diagnose_tiou.py on 195 positives):
- gold mean 3.21s, predicted mean 9.42s: the LLM returns long passages
  (mean 24 tokens, up to 187) despite the "shortest passage" instruction,
  so 60.5% of positives land in the CONTAINS bucket (pred strictly
  contains gold, IoU = gold_len/pred_len).
- systematic bias: start -1.44s early (median -0.33s), end +4.78s late.
- oracle ceilings (word-exact sentences): 1 sentence 0.68, k<=3 runs 0.78.

Method (2 free params, shared across conversations; no sample keywords):
1. split words into sentences on .?! (exact word timings), build all
   contiguous runs of k<=3 sentences (~170 candidates/conversation);
2. keep runs overlapping the raw locate_evidence span (LLM region);
3. pick the run with max content-word F1(question, run);
4. shift start by +DS (DS=0.3, median start bias), no end trim (DE=0.0:
   any end trim hurt on all folds);
5. if raw is None but answer is YES (or no run overlaps), retrieval
   fallback over all runs so YES never yields None.

Gate (offline, cached ASR+LLM, 5-fold CV vs TRUE baseline 0.4161):
- overall 0.4161 -> 0.4875 (delta +0.0714, score 0.6394 -> 0.6822);
- improved folds 5/5; bootstrap over conversations (1000 resamples,
  seed 42) win rate 98.4%; accuracy unchanged (answers untouched).
Scripts: experiments/e1_calibrate.py, e1b_krun.py, e1c_qlex.py, e1d_gate.py.
"""

import re

from medical_evidence import normalize_tokens

START_SHIFT = 0.3
END_TRIM = 0.0
MAX_RUN = 3

_STOP = set(
    "the a an is are was were be been do does did will would should can could "
    "have has had what when where who whom why how it its this that these "
    "those there here and or but of in on at to for with by from as".split()
)


def content_tokens(text):
    return [t for t in normalize_tokens(text) if t not in _STOP]


def _f1(a_toks, b_toks):
    sa = set(a_toks) - _STOP
    sb = set(b_toks) - _STOP
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    precision = inter / len(sb)
    recall = inter / len(sa)
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _overlap(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def build_sentence_runs(words, max_run=MAX_RUN):
    """Sentences from word timings; all contiguous runs of 1..max_run."""
    sentences = []
    current = []
    for word in words:
        current.append(word)
        if re.search(r"[.!?][\"')\]]*$", word["word"]):
            sentences.append(list(current))
            current = []
    if current:
        sentences.append(list(current))

    runs = []
    for k in range(1, max_run + 1):
        for i in range(len(sentences) - k + 1):
            chunk = [w for j in range(i, i + k) for w in sentences[j]]
            text = " ".join(w["word"] for w in chunk)
            runs.append(
                {
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "tokens": normalize_tokens(text),
                }
            )
    return runs


def calibrate_span(raw_span, runs, question_tokens,
                   start_shift=START_SHIFT, end_trim=END_TRIM):
    """Shrink an over-covering raw span to the best question-anchored run."""
    if raw_span is None:
        return None
    overlapping = [
        run for run in runs
        if _overlap(raw_span, (run["start"], run["end"])) > 0
    ]
    if not overlapping:
        start, end = raw_span
    else:
        best = None
        best_score = -1.0
        for run in overlapping:
            score = _f1(question_tokens, run["tokens"])
            if score > best_score:
                best_score = score
                best = run
        if best is None or best_score <= 0:
            start, end = raw_span
        else:
            start, end = best["start"], best["end"]
    start += start_shift
    end -= end_trim
    if end <= start or start < 0:
        return None
    return (start, end)


def retrieval_span(runs, question_tokens,
                   start_shift=START_SHIFT, end_trim=END_TRIM):
    """Pure lexical fallback: best run anywhere, so YES never yields None."""
    best = None
    best_score = -1.0
    for run in runs:
        score = _f1(question_tokens, run["tokens"])
        if score > best_score:
            best_score = score
            best = run
    if best is None or best_score <= 0:
        return None
    start = best["start"] + start_shift
    end = best["end"] - end_trim
    if end <= start or start < 0:
        return None
    return (start, end)
