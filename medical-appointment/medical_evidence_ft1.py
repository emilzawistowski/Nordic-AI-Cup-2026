"""Attempt 9 evidence helpers: v4 re-exports + post-processor A."""

from medical_evidence_v4 import (
    build_numbered_transcript,
    build_sentences,
    calibrate_indexed_span,
    content_tokens,
)

START_SHIFT = 0.3


def span_full_range_shifted(sentences, ids, start_shift=START_SHIFT):
    """Post-processor A: cited range + start shift, no shrink.

    Invalid/empty IDs -> None. Else (min start + shift, max end); if
    end <= start + 0.05 use the unshifted range.
    """
    if not ids or not sentences:
        return None
    n = len(sentences)
    for sid in ids:
        if not isinstance(sid, int) or isinstance(sid, bool):
            return None
        if sid < 0 or sid >= n:
            return None
    start = min(s["start"] for s in (sentences[i] for i in ids)) + start_shift
    end = max(s["end"] for s in (sentences[i] for i in ids))
    if end <= start or start < 0:
        return None
    if end <= start + 0.05:
        start = min(s["start"] for s in (sentences[i] for i in ids))
        if end <= start or start < 0:
            return None
    return (float(start), float(end))


__all__ = [
    "build_numbered_transcript",
    "build_sentences",
    "calibrate_indexed_span",
    "content_tokens",
    "span_full_range_shifted",
]
