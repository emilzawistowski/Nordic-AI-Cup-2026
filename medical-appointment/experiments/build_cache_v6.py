"""Rebuild offline cache for attempt 8 (grid search on v4 shrink).

Per conversation: words (from transcripts/*.json), sentences via v4,
indexed call-1 QA via medical_reasoner_v2 (fresh calls ONLY because the
attempt-6 cache was reverted-deleted; identical production prompt, cached
by hash in experiments/llm_cache_v2/ for reuse).

Output: experiments/cache_v6.json keyed by audio_filename.
No new LLM architecture. No span experiments here.
"""

import hashlib
import json
from pathlib import Path

from utils import group_questions_by_conversation

TRANSCRIPTS_DIR = Path("transcripts")
LLM_CACHE_DIR = Path("experiments/llm_cache_v2")
CACHE_OUT = Path("experiments/cache_v6.json")


def main():
    from medical_evidence_v4 import build_numbered_transcript, build_sentences
    from medical_reasoner_v2 import answer_questions_batch_indexed

    LLM_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    conversations = group_questions_by_conversation()
    print(f"Building v6 cache for {len(conversations)} conversations...")

    # Words straight from cached transcripts (same timings production uses).
    import re

    cache = {}
    for audio_filename, rows in conversations:
        conv_id = audio_filename.replace("conversation_", "").replace(".mp3", "")
        with open(TRANSCRIPTS_DIR / f"{conv_id}.json") as f:
            transcription = json.load(f)

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

        sentences = build_sentences(words)
        numbered = build_numbered_transcript(sentences)
        questions = [r["question"] for r in rows]

        key = hashlib.md5(f"{numbered}\n{questions}".encode()).hexdigest()
        llm_file = LLM_CACHE_DIR / f"{key}.json"
        if llm_file.exists():
            with open(llm_file) as f:
                qa = json.load(f)
            print(f"  {audio_filename}: LLM cache hit")
        else:
            print(f"  {audio_filename}: LLM call...")
            qa = answer_questions_batch_indexed(numbered, questions)
            with open(llm_file, "w") as f:
                json.dump(qa, f)

        cache[audio_filename] = {"words": words, "qa": qa, "rows": rows}

    with open(CACHE_OUT, "w") as f:
        json.dump(cache, f)
    print(f"Wrote {CACHE_OUT}")


if __name__ == "__main__":
    main()
