import csv
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load


MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
TRANSCRIPT_PATH = Path("transcripts/sample_4.json")
CSV_PATH = Path("data/question_train.csv")


def parse_answer(text: str) -> bool:
    normalized = text.strip().upper()

    match = re.search(r"\b(YES|NO)\b", normalized)

    if match is None:
        return False

    return match.group(1) == "YES"


result = json.loads(
    TRANSCRIPT_PATH.read_text(encoding="utf-8")
)

transcript = result.get("text", "").strip()

with CSV_PATH.open(newline="", encoding="utf-8") as handle:
    questions = [
        row
        for row in csv.DictReader(handle)
        if row["transcript_id"] in {
            "sample_4",
            "conversation_sample_4",
        }
    ]

print("Loading:", MODEL_ID)
started = time.perf_counter()
model, tokenizer = load(MODEL_ID)
print("Loaded in:", round(time.perf_counter() - started, 2), "seconds")
print()

correct = 0

for index, row in enumerate(questions, start=1):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict medical transcript verifier. "
                "Determine whether the question is established by the "
                "conversation. A near miss involving a different dose, "
                "duration, medicine, body part, result, or treatment plan "
                "must be answered NO. Information not mentioned must be "
                "answered NO. Use only the transcript. Reply with exactly "
                "YES or NO."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONSULTATION TRANSCRIPT:\n{transcript}\n\n"
                f"QUESTION:\n{row['question']}\n\n"
                "ANSWER:"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    started = time.perf_counter()

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=8,
        verbose=False,
    )

    elapsed = time.perf_counter() - started

    predicted = parse_answer(response)
    expected = row["label"] == "1"
    is_correct = predicted == expected

    correct += int(is_correct)

    print("=" * 88)
    print(f"{index}. {row['question']}")
    print("Expected:", "YES" if expected else "NO")
    print("Predicted:", "YES" if predicted else "NO")
    print("Raw response:", repr(response.strip()))
    print("Correct:", is_correct)
    print("Seconds:", round(elapsed, 3))

print()
print("=== SUMMARY ===")
print("Correct:", correct, "/", len(questions))
print("Accuracy:", round(correct / len(questions), 3))
