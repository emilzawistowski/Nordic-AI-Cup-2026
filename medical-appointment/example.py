import logging
import time

from dtos import ASRQuestionResponseDto
from medical_asr import extract_words, transcribe_audio
from medical_evidence import locate_evidence
from medical_reasoner import answer_question
from utils import audio_duration_seconds, decode_audio


logger = logging.getLogger(__name__)


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

        answers = []
        evidence_start = []
        evidence_end = []

        for question in request.questions:
            try:
                answer, evidence_text = answer_question(
                    transcript,
                    question,
                )

                span = None

                if answer and evidence_text:
                    span = locate_evidence(
                        words,
                        evidence_text,
                    )

                answers.append(bool(answer))

                if span is None:
                    evidence_start.append(None)
                    evidence_end.append(None)
                else:
                    evidence_start.append(float(span[0]))
                    evidence_end.append(float(span[1]))

            except Exception:
                logger.exception(
                    "Question failed: %s",
                    question,
                )

                answers.append(False)
                evidence_start.append(None)
                evidence_end.append(None)

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
