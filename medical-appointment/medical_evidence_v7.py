"""Hybrid evidence v7 (attempt 11): v4 copy + cross-encoder fallback repair.

Verbatim copy of medical_evidence_v4.py, except calibrate_indexed_span is
extended with a deterministic repair layer for the shrink-fallback leak:

- The content-word F1 shrink runs FIRST. If it succeeds, behavior is
  byte-identical to v4 (same span, same fell_back flag).
- If F1 finds no sub-run with F1 > 0 (the fallback trigger), instead of
  returning the full cited range, a pretrained sentence cross-encoder
  (cross-encoder/ms-marco-MiniLM-L-6-v2, CPU) scores every candidate sub-run
  of the cited range (same contiguous-run enumeration as the F1 path)
  against the question TEXT, the best run wins, and the same +START_SHIFT
  applies. A best score below CE_MIN_SCORE (-5.0) or degenerate geometry
  falls back to the full cited range exactly as v4 did.
- Invalid IDs / empty ID list -> (None, False): caller treats as NO,
  identical to v4 (answer side untouched, accuracy delta must be exactly 0).

Instrumentation: every fallback-trigger call appends one dict to REPAIR_LOG
with {fired, ce_score, run_len, n_candidates}. fell_back=False on a
successful repair, True when the full range is returned (as in v4).
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- Copied verbatim from medical_spans_v2.py (attempt 3, gated) ---
from medical_evidence import normalize_tokens

START_SHIFT = 0.3
END_TRIM = 0.0

# --- Attempt-11 repair constants (fixed by design, NOT tuned on data) ---
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_MIN_SCORE = -5.0

# One entry per fallback-trigger call: {fired, ce_score, run_len,
# n_candidates}. Cleared/consumed by the caller (experiments/e7_cv.py).
REPAIR_LOG = []

_ce_model = None


def _get_ce_model():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder

        import torch

        torch.set_num_threads(max(1, torch.get_num_threads()))
        _ce_model = CrossEncoder(CE_MODEL_NAME, device="cpu")
        logger.info("Cross-encoder %s loaded on CPU", CE_MODEL_NAME)
    return _ce_model


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


# --- Sentence segmentation (same .?! rule as spans_v2 / evidence_v3) ---
_SENT_END = re.compile(r"[.!?][\"')\]]*$")


def build_sentences(words):
    """Split word list into sentences. IDs are positions (0-based)."""
    sentences = []
    current = []
    for word in words:
        current.append(word)
        if _SENT_END.search(word["word"]):
            sentences.append(list(current))
            current = []
    if current:
        sentences.append(list(current))

    result = []
    for chunk in sentences:
        result.append(
            {
                "words": chunk,
                "text": " ".join(w["word"] for w in chunk),
                "start": float(chunk[0]["start"]),
                "end": float(chunk[-1]["end"]),
                "tokens": normalize_tokens(
                    " ".join(w["word"] for w in chunk)
                ),
            }
        )
    return result


def build_numbered_transcript(sentences):
    """Render sentences as '[N] text' lines for the LLM prompt."""
    return "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sentences))


def _full_range(sentences, ids):
    """Full (start, end) of cited sentences, or None if IDs invalid/empty."""
    if not ids or not sentences:
        return None
    n = len(sentences)
    for sid in ids:
        if not isinstance(sid, int) or isinstance(sid, bool):
            return None
        if sid < 0 or sid >= n:
            return None
    picked = [sentences[sid] for sid in ids]
    start = min(s["start"] for s in picked)
    end = max(s["end"] for s in picked)
    if end <= start or start < 0:
        return None
    return (float(start), float(end))


def _cite_runs(sentences, ids):
    """All contiguous sub-runs of the cited sentences (v4 enumeration).

    Returns (runs, cited) where each run has start/end/tokens/text/k and
    cited is the sorted unique ID list. Runs are ordered earliest-first,
    then shortest (first-max-wins tie-break, same as v4).
    """
    cited = sorted(set(ids))
    runs = []
    for a in range(len(cited)):
        chunk_words = []
        chunk_toks = []
        chunk_texts = []
        for b in range(a, len(cited)):
            if cited[b] != cited[a] + (b - a):
                break  # keep runs contiguous in transcript order
            s = sentences[cited[b]]
            chunk_words.extend(s["words"])
            chunk_toks.extend(s["tokens"])
            chunk_texts.append(s["text"])
            runs.append(
                {
                    "start": float(chunk_words[0]["start"]),
                    "end": float(chunk_words[-1]["end"]),
                    "tokens": list(chunk_toks),
                    "text": " ".join(chunk_texts),
                    "k": b - a + 1,
                }
            )
    return runs, cited


def calibrate_indexed_span(sentences, ids, question_tokens,
                            start_shift=START_SHIFT, end_trim=END_TRIM,
                            question_text=None):
    """Shrink the cited sentence range to the best question-anchored sub-run.

    Identical to v4 except on the F1-fallback trigger: instead of returning
    the full cited range, a cross-encoder picks the best sub-run (see module
    docstring). question_text is required for the repair; when None, the
    repair cannot run and the v4 full-range fallback is returned.

    Returns (span_or_None, fell_back_bool), same contract as v4.
    """
    full = _full_range(sentences, ids)
    if full is None:
        return None, False

    # Candidates: all contiguous sub-runs of the cited sentences
    # (spans_v2 selection rule, constrained to the cited range).
    runs, _ = _cite_runs(sentences, ids)

    # --- Stage 1 (v4, unchanged): content-word F1 sub-run selection. ---
    best = None
    best_score = -1.0
    for run in runs:
        score = _f1(question_tokens, run["tokens"])
        if score > best_score:
            best_score = score
            best = run

    if best is not None and best_score > 0:
        start = best["start"] + start_shift
        end = best["end"] - end_trim
        if end <= start or start < 0:
            return full, True
        return ((float(start), float(end)), False)

    # --- Stage 2 (attempt 11): cross-encoder repair on F1 failure. ---
    if question_text is None:
        REPAIR_LOG.append(
            {"fired": False, "ce_score": None, "run_len": None,
             "n_candidates": len(runs), "reason": "no-question-text"}
        )
        return full, True

    model = _get_ce_model()
    pairs = [(question_text, run["text"]) for run in runs]
    ce_scores = [float(s) for s in model.predict(pairs)]
    top = int(max(range(len(runs)), key=lambda i: ce_scores[i]))
    top_score = ce_scores[top]

    if top_score < CE_MIN_SCORE:
        logger.info(
            "CE_REPAIR_SKIP score=%.2f below %.1f -> full cited range",
            top_score, CE_MIN_SCORE,
        )
        REPAIR_LOG.append(
            {"fired": False, "ce_score": top_score, "run_len": None,
             "n_candidates": len(runs), "reason": "below-threshold"}
        )
        return full, True

    chosen = runs[top]
    start = chosen["start"] + start_shift
    end = chosen["end"] - end_trim
    if end <= start or start < 0:
        REPAIR_LOG.append(
            {"fired": False, "ce_score": top_score, "run_len": None,
             "n_candidates": len(runs), "reason": "degenerate-geometry"}
        )
        return full, True

    logger.info(
        "CE_REPAIR_FIRED score=%.2f run_len=%d n_cand=%d",
        top_score, chosen["k"], len(runs),
    )
    REPAIR_LOG.append(
        {"fired": True, "ce_score": top_score, "run_len": chosen["k"],
         "n_candidates": len(runs), "reason": None}
    )
    return ((float(start), float(end)), False)
