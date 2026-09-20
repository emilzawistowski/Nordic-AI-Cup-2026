"""Thin pass-through (attempt 10): geometry/shrink stage UNCHANGED.

Re-exports from medical_evidence_v4 by import only — no copied or forked
logic. The MBR experiment fuses reasoner votes; calibrate_indexed_span,
sentence building, and START_SHIFT/END_TRIM stay exactly as in production.
"""

from medical_evidence_v4 import (  # noqa: F401
    END_TRIM,
    START_SHIFT,
    build_numbered_transcript,
    build_sentences,
    calibrate_indexed_span,
    content_tokens,
)
