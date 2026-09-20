"""Experimental entry point — cross-encoder fallback repair (attempt 11).

Copy of example.py with imports switched to medical_reasoner_v7 (unchanged
reasoner) and medical_evidence_v7 (v4 shrink + cross-encoder repair on the
F1-fallback trigger). Passes the question text through for the repair stage.

NOT wired to api.py. Do NOT promote without manual go-ahead.
"""

import logging
import time

from dtos import ASRQuestionResponseDto
from medical_asr import extract_words, transcribe_audio
from medical_evidence_v7 import (
    build_numbered_transcript,
    build_sentences,
    calibrate_indexed_span,
    content_tokens,
)
from medical_reasoner_v7 import answer_questions_batch_indexed
from utils import audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

# Model preload at import (no warm-up period; the first request is slowest).
# Failures here must not break import.
try:
    from medical_reasoner_v7 import get_model as _get_reasoning_model

    _model, _tokenizer = _get_reasoning_model()
    logger.info("Reasoning model (v7) preloaded at import")
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
        qa_results = answer_questions_batch_indexed(numbered, request.questions)
        if qa_results is None or len(qa_results) != count:
            raise ValueError(
                f"LLM returned {0 if qa_results is None else len(qa_results)} "
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
                    sentences, ids, question_tokens,
                    question_text=request.questions[idx],
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
