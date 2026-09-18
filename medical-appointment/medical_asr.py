import tempfile
import time
from pathlib import Path

import mlx_whisper


WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe_audio(audio_bytes, audio_filename):
    suffix = Path(audio_filename).suffix or ".mp3"
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = Path(temporary_file.name)

        started = time.perf_counter()

        result = mlx_whisper.transcribe(
            str(temporary_path),
            path_or_hf_repo=WHISPER_MODEL,
            language="en",
            task="transcribe",
            word_timestamps=True,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=True,
        )

        result["_elapsed_seconds"] = time.perf_counter() - started
        return result

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_words(transcription):
    words = []

    for segment in transcription.get("segments", []):
        for word in segment.get("words", []):
            text = str(word.get("word", "")).strip()

            if not text:
                continue

            words.append(
                {
                    "word": text,
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                }
            )

    return words
