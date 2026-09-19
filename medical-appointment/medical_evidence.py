import re
from difflib import SequenceMatcher

def normalize_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def extract_numbers(tokens):
    numbers = []
    num_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    for t in tokens:
        if t in num_words:
            numbers.append(float(num_words[t]))
        else:
            try:
                numbers.append(float(t))
            except ValueError:
                pass
    return numbers

def extract_negations(tokens):
    neg_words = {"not", "no", "never", "without", "none", "neither", "nor", "normal", "abnormal", "denies", "denied"}
    return set(tokens) & neg_words

def locate_evidence(words, evidence_text):
    target_tokens = normalize_tokens(evidence_text)
    if not target_tokens or not words:
        return None

    flattened = []
    for word_index, word in enumerate(words):
        for token in normalize_tokens(word["word"]):
            flattened.append({
                "token": token,
                "word_index": word_index,
            })

    transcript_tokens = [item["token"] for item in flattened]
    target_length = len(target_tokens)
    if target_length > len(transcript_tokens):
        return None

    target_numbers = extract_numbers(target_tokens)
    target_negations = extract_negations(target_tokens)

    # 1. Exact match search across candidate windows
    exact_candidates = []
    for start in range(len(transcript_tokens) - target_length + 1):
        end = start + target_length
        if transcript_tokens[start:end] == target_tokens:
            first_word = flattened[start]["word_index"]
            last_word = flattened[end - 1]["word_index"]
            exact_candidates.append((first_word, last_word))

    if exact_candidates:
        # Choose candidate with optimal word alignment
        first_word, last_word = exact_candidates[0]
        return (
            max(0.0, words[first_word]["start"] - 0.05),
            words[last_word]["end"] + 0.05,
        )

    # 2. Candidate region selection with guarded numeric/negation alignment
    target_text = " ".join(target_tokens)
    target_set = set(target_tokens)

    best_score = 0.0
    best_span = None

    minimum_length = max(1, target_length - 4)
    maximum_length = min(len(transcript_tokens), target_length + 6)

    for window_length in range(minimum_length, maximum_length + 1):
        for start in range(len(transcript_tokens) - window_length + 1):
            end = start + window_length
            candidate_tokens = transcript_tokens[start:end]

            # Guarded numeric check: candidate must contain target numbers if target has specific numeric values
            if target_numbers:
                cand_numbers = extract_numbers(candidate_tokens)
                if set(target_numbers) != set(cand_numbers):
                    continue

            # Guarded negation check
            cand_negations = extract_negations(candidate_tokens)
            if target_negations and not cand_negations:
                continue

            candidate_text = " ".join(candidate_tokens)
            sequence_score = SequenceMatcher(None, target_text, candidate_text).ratio()
            union = target_set | set(candidate_tokens)
            overlap_score = (len(target_set & set(candidate_tokens)) / len(union)) if union else 0.0
            score = 0.70 * sequence_score + 0.30 * overlap_score

            if score > best_score:
                first_word = flattened[start]["word_index"]
                last_word = flattened[end - 1]["word_index"]
                best_score = score
                best_span = (
                    max(0.0, words[first_word]["start"] - 0.05),
                    words[last_word]["end"] + 0.05,
                )

    if best_score < 0.45:
        return None

    return best_span
