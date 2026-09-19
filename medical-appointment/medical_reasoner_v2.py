"""Index-based reasoner (attempts 4-5, idea #2).

Copy of medical_reasoner.py with an index-only citation prompt.

The transcript MUST be passed in pre-numbered, one sentence per line:
    [0] first sentence ...
    [1] second sentence ...
Sentence IDs are the only allowed citation format. Free-text quotes are
forbidden, so span mapping is a deterministic lookup (see
medical_evidence_v3/v4.py) instead of fuzzy matching.

Output contract per question (one line each):
    1. YES [2,3]
    2. NO
A YES with an empty/missing ID list is parsed as (True, []) and the caller
treats it as NO. Out-of-range IDs are also the caller's call (NO + warning).
"""

import logging
import re

from mlx_lm import generate, load

REASONING_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_model = None
_tokenizer = None

logger = logging.getLogger(__name__)


def get_model():
    global _model
    global _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load(REASONING_MODEL)
    return _model, _tokenizer


def parse_indexed_response(text, num_questions):
    """Parse 'N. YES [a,b]' / 'N. NO' lines -> [(bool, [ids])]."""
    lines = text.strip().split("\n")
    results = []
    for i in range(1, num_questions + 1):
        found = False
        for line in lines:
            stripped = line.strip()
            pattern = rf"^\s*{i}\s*[\.\)]\s*(YES|NO)\b(.*)"
            match = re.search(pattern, stripped, flags=re.I)
            if match:
                ans = match.group(1).upper() == "YES"
                ids = []
                if ans:
                    bracket = re.search(r"\[([0-9,\s]*)\]", match.group(2))
                    if bracket and bracket.group(1).strip():
                        try:
                            ids = [
                                int(x)
                                for x in bracket.group(1).split(",")
                                if x.strip() != ""
                            ]
                        except ValueError:
                            ids = []
                    else:
                        # Lenient fallback: bare ints after YES
                        # (e.g. "YES 2,3" without brackets).
                        bare = re.findall(r"\d+", match.group(2))
                        ids = [int(x) for x in bare]
                results.append((ans, ids))
                found = True
                break
        if not found:
            # Lenient fallback: any line mentioning this number with YES.
            for line in lines:
                if re.search(rf"(^|\D){i}(\D|$)", line):
                    ans = "YES" in line.upper()
                    ids = []
                    if ans:
                        for grp in re.findall(r"\[([0-9,\s]*)\]", line):
                            for x in grp.split(","):
                                x = x.strip()
                                if x != "":
                                    try:
                                        ids.append(int(x))
                                    except ValueError:
                                        pass
                    results.append((ans, ids))
                    found = True
                    break
        if not found:
            results.append((False, []))
    return results


def answer_questions_batch_indexed(numbered_transcript, questions):
    """Answer questions against a pre-numbered transcript.

    Args:
        numbered_transcript: string with one "[N] sentence" per line.
        questions: list of question strings.

    Returns:
        list of (answer_bool, sentence_id_list) per question.
    """
    model, tokenizer = get_model()

    q_str = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

    messages = [
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
