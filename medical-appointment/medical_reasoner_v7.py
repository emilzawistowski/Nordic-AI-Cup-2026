"""Thin pass-through wrapper (attempt 11): reasoner UNCHANGED.

Re-exports the exact functions from medical_reasoner_v2.py. No behavior
change — the attempt-11 signal is purely the v7 evidence repair stage.
"""

from medical_reasoner_v2 import (  # noqa: F401
    REASONING_MODEL,
    answer_questions_batch_indexed,
    get_model,
    parse_indexed_response,
)
