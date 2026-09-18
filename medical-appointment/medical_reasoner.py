import re

from mlx_lm import generate, load


REASONING_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_model = None
_tokenizer = None


def get_model():
    global _model
    global _tokenizer

    if _model is None or _tokenizer is None:
        _model, _tokenizer = load(REASONING_MODEL)

    return _model, _tokenizer


def parse_response(text):
    cleaned = text.strip()

    match = re.search(
        r"(?is)\b(YES|NO)\b\s*\|\s*(.*)",
        cleaned,
    )

    if match is None:
        match = re.search(r"(?i)\b(YES|NO)\b", cleaned)

        if match is None:
            return False, ""

        return match.group(1).upper() == "YES", ""

    answer = match.group(1).upper() == "YES"
    evidence = match.group(2).strip()

    evidence = evidence.strip('"')
    evidence = evidence.strip("'")
    evidence = evidence.strip()

    if not answer:
        evidence = ""

    return answer, evidence


def answer_question(transcript, question):
    model, tokenizer = get_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict medical transcript verifier. "
                "Determine whether the question is established by the "
                "consultation. Answer NO when information is absent. "
                "Answer NO for a near miss involving a different medicine, "
                "dose, unit, duration, frequency, body part, result, symptom, "
                "test result, or treatment plan. Medical paraphrases may be "
                "equivalent. Use only the supplied transcript. "
                "For YES, copy the shortest complete supporting passage "
                "exactly from the transcript. Do not paraphrase it. "
                "For NO, provide no evidence. "
                "Reply in exactly one of these formats: "
                "YES | exact copied passage "
                "or "
                "NO |"
            ),
        },
        {
            "role": "user",
            "content": (
                "CONSULTATION TRANSCRIPT:\n"
                + transcript
                + "\n\nQUESTION:\n"
                + question
                + "\n\nANSWER:"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=120,
        verbose=False,
    )

    return parse_response(response)
