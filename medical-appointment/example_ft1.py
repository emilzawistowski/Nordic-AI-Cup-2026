"""PLACEHOLDER — attempt 9. Wiring (POSTPROC) is set from the gate result.

PASS branch only: POSTPROC = winning variant ("A" or "B"), served via
api_ft1.py:9058 for a wiring health check. Do NOT point example.py here.
"""

import logging
import time

from dtos import ASRQuestionResponseDto
from medical_asr import extract_words, transcribe_audio
from medical_evidence_ft1 import (
    build_numbered_transcript,
    build_sentences,
    calibrate_indexed_span,
    content_tokens,
    span_full_range_shifted,
)
from medical_reasoner_ft1 import answer_questions_batch_indexed
from utils import audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

POSTPROC = "B"  # set from gate result on PASS ("A" or "B")

try:
    from medical_reasoner_ft1 import get_model as _get_reasoning_model

    _model, _tokenizer = _get_reasoning_model()
    logger.info("Reasoning model (ft1) preloaded at import")
except Exception:
    logger.exception("Reasoning model preload failed (will retry per request)")


def predict(request):
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
        transcription = transcribe_audio(audio_bytes, request.audio_filename)
        words = extract_words(transcription)
    except Exception:
        logger.exception("ASR stage failed for %s", request.audio_filename)
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

    try:
        qa_results = answer_questions_batch_indexed(numbered, request.questions)
        if qa_results is None or len(qa_results) != count:
            raise ValueError("LLM returned wrong count")
    except Exception:
        logger.exception("LLM stage failed for %s", request.audio_filename)
        return ASRQuestionResponseDto(
            answers=[False] * count,
            evidence_start=[None] * count,
            evidence_end=[None] * count,
        )

    answers = []
    evidence_start = []
    evidence_end = []
    for idx, (answer, ids) in enumerate(qa_results):
        span = None
        final = bool(answer)
        if final:
            qt = content_tokens(request.questions[idx])
            if POSTPROC == "A":
                span = span_full_range_shifted(sentences, ids)
            else:
                span, _ = calibrate_indexed_span(sentences, ids, qt)
            if span is None:
                final = False
        answers.append(final)
        evidence_start.append(float(span[0]) if span else None)
        evidence_end.append(float(span[1]) if span else None)

    logger.info("Completed %s in %.2f seconds", request.audio_filename,
                time.perf_counter() - started)
    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )
