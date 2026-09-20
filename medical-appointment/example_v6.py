"""Experimental entry point — MBR vote fusion (attempt 10, NOT production).

Mirrors example.py but replaces the single deterministic reasoner call with
5 sampled runs (temp=0.7, top_p=0.9, seeds from medical_reasoner_v6) fused
via fuse_votes, then the EXISTING UNCHANGED v4 calibrate_indexed_span.

NOT wired to api.py. Do NOT promote without manual go-ahead.
"""

import logging
import time

from dtos import ASRQuestionResponseDto
from medical_asr import extract_words, transcribe_audio
from medical_evidence_v6 import (
    build_numbered_transcript,
    build_sentences,
    calibrate_indexed_span,
    content_tokens,
)
from medical_reasoner_v6 import (
    N_VOTES,
    SAMPLE_TEMP,
    SAMPLE_TOP_P,
    VOTE_SEEDS,
    answer_questions_batch_indexed_sampled,
    fuse_votes,
)
from utils import audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

# Tunable free parameter (swept in experiments/e6_mbr_cv.py, not hardcoded
# as a guess here beyond the plain-majority default).
YES_THRESHOLD = 3

# Model preload at import (no warm-up period; the first request is slowest).
try:
    from medical_reasoner_v2 import get_model as _get_reasoning_model

    _model, _tokenizer = _get_reasoning_model()
    logger.info("Reasoning model (v2) preloaded at import")
except Exception:
    logger.exception("Reasoning model preload failed (will retry per request)")


def predict(request, yes_threshold=YES_THRESHOLD):
    started = time.perf_counter()
    count = len(request.questions)

    try:
        audio_bytes = decode_audio(request.audio_base64)
        duration = audio_duration_seconds(audio_bytes)

        logger.info(
            "%s, duration %.1f seconds, %d questions",
            request.audio_filename,
            duration if duration is not None else -1.0,
            count,
        )

        transcription = transcribe_audio(
            audio_bytes,
            request.audio_filename,
        )
        words = extract_words(transcription)
    except Exception:
        logger.exception(
            "ASR stage failed for %s",
            request.audio_filename,
        )
        return ASRQuestionResponseDto(
            answers=[False] * count,
            evidence_start=[None] * count,
            evidence_end=[None] * count,
        )

    try:
        sentences = build_sentences(words)
    except Exception:
        logger.exception("Sentence build failed for %s", request.audio_filename)
        sentences = []

    numbered = build_numbered_transcript(sentences)

    # LLM stage isolated: a failure after successful ASR returns all-False.
    try:
        runs = []
        for seed in VOTE_SEEDS:
            _, parsed = answer_questions_batch_indexed_sampled(
                numbered,
                request.questions,
                temp=SAMPLE_TEMP,
                top_p=SAMPLE_TOP_P,
                seed=seed,
            )
            runs.append(parsed)
        if any(r is None or len(r) != count for r in runs):
            raise ValueError(
                f"Sampled LLM returned wrong shape for {count} questions"
            )
        qa_results = fuse_votes(runs, yes_threshold=yes_threshold)
        if qa_results is None or len(qa_results) != count:
            raise ValueError(
                f"Fused LLM returned {0 if qa_results is None else len(qa_results)} "
                f"results for {count} questions"
            )
    except Exception:
        logger.exception(
            "LLM stage failed for %s",
            request.audio_filename,
        )
        return ASRQuestionResponseDto(
            answers=[False] * count,
            evidence_start=[None] * count,
            evidence_end=[None] * count,
        )

    try:
        answers = []
        evidence_start = []
        evidence_end = []

        for idx, (answer, ids) in enumerate(qa_results):
            span = None
            final_answer = bool(answer)
            if final_answer:
                question_tokens = content_tokens(request.questions[idx])
                span, fell_back = calibrate_indexed_span(
                    sentences, ids, question_tokens
                )
                if span is None:
                    logger.warning(
                        "INVALID_SENTENCE_ID %s q%d: ids=%r out of range "
                        "(n_sent=%d) -> treated as NO",
                        request.audio_filename,
                        idx,
                        ids,
                        len(sentences),
                    )
                    final_answer = False
                elif fell_back:
                    logger.info(
                        "SHRINK_FALLBACK %s q%d: ids=%r full cited range used",
                        request.audio_filename,
                        idx,
                        ids,
                    )

            answers.append(final_answer)
            if span is None:
                evidence_start.append(None)
                evidence_end.append(None)
            else:
                evidence_start.append(float(span[0]))
                evidence_end.append(float(span[1]))

        logger.info(
            "Completed %s in %.2f seconds",
            request.audio_filename,
            time.perf_counter() - started,
        )

        return ASRQuestionResponseDto(
            answers=answers,
            evidence_start=evidence_start,
            evidence_end=evidence_end,
        )

    except Exception:
        logger.exception(
            "Pipeline failed for %s",
            request.audio_filename,
        )

        return ASRQuestionResponseDto(
            answers=[False] * count,
            evidence_start=[None] * count,
            evidence_end=[None] * count,
        )
