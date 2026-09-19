import logging
import time

from dtos import ASRQuestionResponseDto
from medical_asr import extract_words, transcribe_audio
from medical_evidence import locate_evidence
from medical_reasoner import answer_questions_batch
from medical_spans_v2 import (
    build_sentence_runs,
    calibrate_span,
    content_tokens,
    retrieval_span,
)
from utils import audio_duration_seconds, decode_audio

logger = logging.getLogger(__name__)

# Phase 3 hardening: load + exercise models at import (no warm-up period;
# the first request is the slowest). Failures here must not break import.
try:
    from medical_reasoner import get_model as _get_reasoning_model

    _model, _tokenizer = _get_reasoning_model()
    logger.info("Reasoning model preloaded at import")
except Exception:
    logger.exception("Reasoning model preload failed (will retry per request)")

_FALLBACK_YES_THRESHOLD = 0.3

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
        transcript = transcription.get("text", "").strip()
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
        runs = build_sentence_runs(words)
    except Exception:
        logger.exception("Sentence-run build failed for %s", request.audio_filename)
        runs = []

    # LLM stage isolated: a failure after successful ASR still returns
    # retrieval-based answers/spans instead of all-False.
    try:
        # Structured batch generation of all 10 questions at once
        qa_results = answer_questions_batch(transcript, request.questions)
        if qa_results is None or len(qa_results) != count:
            raise ValueError(
                f"LLM returned {0 if qa_results is None else len(qa_results)} "
                f"results for {count} questions"
            )
    except Exception:
        logger.exception(
            "LLM stage failed for %s, using retrieval fallback",
            request.audio_filename,
        )
        qa_results = None

    try:
        answers = []
        evidence_start = []
        evidence_end = []

        if qa_results is None:
            from medical_spans_v2 import _f1 as _run_f1

            for question in request.questions:
                question_tokens = content_tokens(question)
                best_score = -1.0
                best_run = None
                for run in runs:
                    score = _run_f1(question_tokens, run["tokens"])
                    if score > best_score:
                        best_score = score
                        best_run = run
                answer = bool(best_run is not None and best_score > _FALLBACK_YES_THRESHOLD)
                answers.append(answer)
                span = None
                if answer and best_run is not None:
                    span = retrieval_span(runs, question_tokens)
                if span is None:
                    evidence_start.append(None)
                    evidence_end.append(None)
                else:
                    evidence_start.append(float(span[0]))
                    evidence_end.append(float(span[1]))
        else:
            for idx, (answer, evidence_text) in enumerate(qa_results):
                span = None
                try:
                    question_tokens = content_tokens(request.questions[idx])
                    if answer and evidence_text:
                        raw = locate_evidence(words, evidence_text)
                        span = calibrate_span(raw, runs, question_tokens)
                        if span is None:
                            # YES never yields None: retrieval fallback
                            span = retrieval_span(runs, question_tokens)
                except Exception:
                    logger.exception(
                        "Span stage failed for %s q%d",
                        request.audio_filename,
                        idx,
                    )
                    span = None

                answers.append(bool(answer))
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
