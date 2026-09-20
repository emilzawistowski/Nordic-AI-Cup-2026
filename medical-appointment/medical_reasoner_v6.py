"""MBR/majority-vote reasoner (attempt 10).

Wraps medical_reasoner_v2's indexed batched QA prompt (byte-identical prompt)
but samples with temperature instead of greedy decoding, so one conversation
yields N diverse parses. Aggregation happens in fuse_votes below; geometry
(medical_evidence_v4) is untouched.

Sampling: mlx_lm.sample_utils.make_sampler(temp=0.7, top_p=0.9, top_k
default), independent random seed per call via mx.random.seed.
"""

import logging

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from medical_reasoner_v2 import get_model, parse_indexed_response

logger = logging.getLogger(__name__)

SAMPLE_TEMP = 0.7
SAMPLE_TOP_P = 0.9
# Independent seeds, one per vote run (fixed so experiments/e6_cache/ is
# reproducible without re-running MLX inference).
VOTE_SEEDS = (101, 102, 103, 104, 105)
N_VOTES = len(VOTE_SEEDS)


def _build_prompt(tokenizer, numbered_transcript, questions):
    """Prompt identical to medical_reasoner_v2.answer_questions_batch_indexed."""
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

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def answer_questions_batch_indexed_sampled(
    numbered_transcript, questions, temp=SAMPLE_TEMP, top_p=SAMPLE_TOP_P, seed=0
):
    """One sampled QA pass over the SAME numbered transcript.

    Returns (raw_text, parsed) where parsed is a list of (bool, [ids]) per
    question, using medical_reasoner_v2.parse_indexed_response.
    """
    model, tokenizer = get_model()
    prompt = _build_prompt(tokenizer, numbered_transcript, questions)

    sampler = make_sampler(temp=temp, top_p=top_p)
    mx.random.seed(seed)
    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=600,
        sampler=sampler,
        verbose=False,
    )
    return raw, parse_indexed_response(raw, len(questions))


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fuse_votes(list_of_runs, yes_threshold=3):
    """Fuse N sampled runs into one (bool, [ids]) per question.

    Args:
        list_of_runs: list of runs; each run is a list of (bool, [ids]) per
            question, or None if that run failed outright (reduces N).
        yes_threshold: minimum YES votes needed to answer YES.

    Rules:
        1. YES/NO: YES iff yes_count >= yes_threshold, else NO (ties and
           short quorums from failed runs default to NO).
        2. ID set for YES: among the runs that voted YES, the medoid — the
           set with the highest average Jaccard similarity to the other YES
           runs' sets (NOT a union, NOT the first run). Single YES vote ->
           its set as-is.
    """
    valid = [r for r in list_of_runs if r is not None]
    if not valid:
        return []
    n_questions = len(valid[0])
    fused = []
    for qi in range(n_questions):
        yes_sets = [
            list(run[qi][1]) for run in valid if bool(run[qi][0])
        ]
        yes_count = len(yes_sets)
        if yes_count < yes_threshold:
            fused.append((False, []))
            continue
        if len(yes_sets) == 1:
            fused.append((True, list(yes_sets[0])))
            continue
        best_idx, best_avg = 0, -1.0
        for i, cand in enumerate(yes_sets):
            sims = [_jaccard(cand, other) for j, other in enumerate(yes_sets) if j != i]
            avg = sum(sims) / len(sims) if sims else 0.0
            if avg > best_avg:
                best_avg, best_idx = avg, i
        fused.append((True, list(yes_sets[best_idx])))
    return fused
