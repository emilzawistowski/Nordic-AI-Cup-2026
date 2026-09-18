import re
from difflib import SequenceMatcher


def normalize_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def locate_evidence(words, evidence_text):
    target_tokens = normalize_tokens(evidence_text)

    if not target_tokens or not words:
        return None

    flattened = []

    for word_index, word in enumerate(words):
        for token in normalize_tokens(word["word"]):
            flattened.append(
                {
                    "token": token,
                    "word_index": word_index,
                }
            )

    transcript_tokens = [item["token"] for item in flattened]
    target_length = len(target_tokens)

    if target_length > len(transcript_tokens):
        return None

    for start in range(len(transcript_tokens) - target_length + 1):
        end = start + target_length

        if transcript_tokens[start:end] == target_tokens:
            first_word = flattened[start]["word_index"]
            last_word = flattened[end - 1]["word_index"]

            return (
                max(0.0, words[first_word]["start"] - 0.08),
                words[last_word]["end"] + 0.08,
            )

    target_text = " ".join(target_tokens)
    target_set = set(target_tokens)

    best_score = 0.0
    best_span = None

    minimum_length = max(1, target_length - 4)
    maximum_length = min(
        len(transcript_tokens),
        target_length + 6,
    )

    for window_length in range(minimum_length, maximum_length + 1):
        for start in range(
            len(transcript_tokens) - window_length + 1
        ):
            end = start + window_length
            candidate_tokens = transcript_tokens[start:end]
            candidate_text = " ".join(candidate_tokens)

            sequence_score = SequenceMatcher(
                None,
                target_text,
                candidate_text,
            ).ratio()

            union = target_set | set(candidate_tokens)

            if union:
                overlap_score = (
                    len(target_set & set(candidate_tokens)) / len(union)
                )
            else:
                overlap_score = 0.0

            score = (
                0.72 * sequence_score
                + 0.28 * overlap_score
            )

            if score > best_score:
                first_word = flattened[start]["word_index"]
                last_word = flattened[end - 1]["word_index"]

                best_score = score
                best_span = (
                    max(0.0, words[first_word]["start"] - 0.10),
                    words[last_word]["end"] + 0.10,
                )

    if best_score < 0.50:
        return None

    return best_span
