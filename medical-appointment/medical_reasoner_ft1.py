"""Attempt 9 reasoner: v2-identical prompts, optional LoRA adapter.

build_messages(numbered, questions) uses v2's system/user strings copied
byte-for-byte. get_model(adapter_path) loads the SAME 4-bit Qwen3-4B with
mlx_lm.load(REASONING_MODEL, adapter_path=...) unfused; default adapter =
env MEDFT1_ADAPTER or models/medft1/full.
"""

import logging
import os

from mlx_lm import generate, load

from medical_reasoner_v2 import parse_indexed_response

REASONING_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_ADAPTER = os.environ.get("MEDFT1_ADAPTER", "models/medft1/full")

logger = logging.getLogger(__name__)

_model_cache = {}


def build_messages(numbered_transcript, questions):
    q_str = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
    return [
        {
            "role": "system",
            "content": (
                "You are a strict medical transcript verifier.\n"
                "The transcript is numbered, one sentence per line: "
                "[0] ..., [1] ..., etc.\n"
                "For each question, determine if it is established by the "
                "transcript.\n"
                "Answer NO when information is absent or for near misses "
                "(different drug, dose, unit, duration, frequency, result).\n"
                "For YES, cite ONLY the sentence IDs that support it, e.g. "
                "YES [2,3]. A single sentence is YES [5].\n"
                "Sentence IDs are the ONLY allowed citation format. "
                "Free-text quotes are FORBIDDEN — never copy transcript text.\n"
                "Output each answer on a new line formatted exactly as:\n"
                "1. YES [2,3]\n"
                "2. NO\n"
                "..."
            ),
        },
        {
            "role": "user",
            "content": (
                f"NUMBERED TRANSCRIPT (one sentence per line):\n"
                f"{numbered_transcript}\n\n"
                f"QUESTIONS:\n{q_str}\n\n"
                "ANSWERS:"
            ),
        },
    ]


def get_model(adapter_path=None):
    key = adapter_path or "__base__"
    if key not in _model_cache:
        if adapter_path:
            _model_cache[key] = load(REASONING_MODEL, adapter_path=adapter_path)
        else:
            _model_cache[key] = load(REASONING_MODEL)
    return _model_cache[key]


def answer_questions_batch_indexed(numbered_transcript, questions, adapter_path=None):
    if adapter_path is None:
        adapter_path = DEFAULT_ADAPTER if os.path.isdir(DEFAULT_ADAPTER) else None
    model, tokenizer = get_model(adapter_path)
    messages = build_messages(numbered_transcript, questions)
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=600,
        verbose=False,
    )
    return parse_indexed_response(response, len(questions))
