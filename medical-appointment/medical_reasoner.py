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

def parse_batch_response(text, num_questions):
    lines = text.strip().split("\n")
    results = []
    
    for i in range(1, num_questions + 1):
        found = False
        for line in lines:
            pattern = rf"^\s*{i}\s*[\.\)]\s*(YES|NO)\s*\|\s*(.*)"
            match = re.search(pattern, line.strip(), flags=re.I)
            if match:
                ans = match.group(1).upper() == "YES"
                ev = match.group(2).strip().strip('"\'') if ans else ""
                results.append((ans, ev))
                found = True
                break
        if not found:
            # Fallback simple search
            for line in lines:
                if f"{i}." in line or f"{i})" in line:
                    ans = "YES" in line.upper()
                    results.append((ans, ""))
                    found = True
                    break
        if not found:
            results.append((False, ""))
            
    return results

def answer_questions_batch(transcript, questions):
    model, tokenizer = get_model()
    
    q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict medical transcript verifier.\n"
                "For each question, determine if it is established by the transcript.\n"
                "Answer NO when information is absent or for near misses (different drug, dose, unit, duration, frequency, result).\n"
                "For YES, copy the exact shortest supporting passage.\n"
                "Output each answer on a new line formatted as:\n"
                "1. YES | exact passage\n"
                "2. NO |\n"
                "..."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONSULTATION TRANSCRIPT:\n{transcript}\n\n"
                f"QUESTIONS:\n{q_str}\n\n"
                "ANSWERS:"
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
        max_tokens=600,
        verbose=False,
    )

    return parse_batch_response(response, len(questions))

def answer_question(transcript, question):
    res = answer_questions_batch(transcript, [question])
    return res[0]
