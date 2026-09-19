"""Hybrid evidence v4 (attempt 5): ID anchoring + shrink (idea #2b).

Attempt 4 showed index lookup alone (tIoU 0.438) loses to fuzzy+shrink
(0.4875) because full cited sentences over-cover. This keeps the
deterministic ID->timestamp anchoring but adds the shrink back.

Method (adapted from medical_spans_v2.py, same params):
1. cited sentence IDs -> full word range (start/end from Whisper timings);
2. WITHIN that range, consider all contiguous sentence sub-runs and pick
   the one with max content-word F1(question, run) — the exact selection
   rule from spans_v2.calibrate_span, except candidates are constrained to
   the cited range instead of "runs overlapping a fuzzy span";
3. shift start by +START_SHIFT (0.3, median start bias), no end trim;
4. if no sub-run has F1 > 0 or geometry is degenerate, fall back to the
   full cited range (caller counts SHRINK_FALLBACK).

Invalid IDs (out of range) or empty ID list -> None (caller treats as NO).
No SequenceMatcher anywhere.
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- Copied verbatim from medical_spans_v2.py (attempt 3, gated) ---
from medical_evidence import normalize_tokens

START_SHIFT = 0.3
END_TRIM = 0.0

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


def calibrate_indexed_span(sentences, ids, question_tokens,
                            start_shift=START_SHIFT, end_trim=END_TRIM):
    """Shrink the cited sentence range to the best question-anchored sub-run.

    Returns (span_or_None, fell_back_bool). fell_back=True means the full
    cited range is returned (no sub-run had F1 > 0, or geometry degenerate).
    span None means invalid IDs (caller treats question as NO).
    """
    full = _full_range(sentences, ids)
    if full is None:
        return None, False

    # Candidates: all contiguous sub-runs of the cited sentences
    # (spans_v2 selection rule, constrained to the cited range).
    cited = sorted(set(ids))
    runs = []
    for a in range(len(cited)):
        chunk_words = []
        chunk_toks = []
        for b in range(a, len(cited)):
            if cited[b] != cited[a] + (b - a):
                break  # keep runs contiguous in transcript order
            s = sentences[cited[b]]
            chunk_words.extend(s["words"])
            chunk_toks.extend(s["tokens"])
            runs.append(
                {
                    "start": float(chunk_words[0]["start"]),
                    "end": float(chunk_words[-1]["end"]),
                    "tokens": list(chunk_toks),
                }
            )

    best = None
    best_score = -1.0
    for run in runs:
        score = _f1(question_tokens, run["tokens"])
        if score > best_score:
            best_score = score
            best = run

    if best is None or best_score <= 0:
        return full, True

    start = best["start"] + start_shift
    end = best["end"] - end_trim
    if end <= start or start < 0:
        return full, True
    return ((float(start), float(end)), False)
