import json
import re
from pathlib import Path


TRANSCRIPT_PATH = Path("transcripts/sample_4.json")
WINDOW_WORDS = 16
WINDOW_STEP = 4


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return " ".join(text.split())


with TRANSCRIPT_PATH.open(encoding="utf-8") as file:
    result = json.load(file)

words = [
    {
        "text": word["word"].strip(),
        "start": float(word["start"]),
        "end": float(word["end"]),
    }
    for segment in result.get("segments", [])
    for word in segment.get("words", [])
    if word.get("word", "").strip()
]

windows = []

for start_index in range(0, len(words), WINDOW_STEP):
    selected = words[start_index:start_index + WINDOW_WORDS]

    if not selected:
        continue

    text = " ".join(word["text"] for word in selected)

    windows.append(
        {
            "start": selected[0]["start"],
            "end": selected[-1]["end"],
            "text": text,
            "normalized": normalize(text),
        }
    )

questions = [
    "Did the patient attend for an annual asthma follow-up?",
    "Is the heart examination without abnormal findings?",
    "Was the patient listened to with a stethoscope?",
    "Were the lungs found to be normal on auscultation?",
    "Are the patient's asthma findings stable at this visit?",
]

keywords = {
    "annual",
    "asthma",
    "follow",
    "heart",
    "normal",
    "chest",
    "listen",
    "lungs",
    "stable",
    "examination",
    "stethoscope",
    "auscultation",
}

print("Words:", len(words))
print("Windows:", len(windows))
print()

for question in questions:
    question_terms = set(normalize(question).split())
    useful_terms = question_terms & keywords

    ranked = []

    for window in windows:
        window_terms = set(window["normalized"].split())
        overlap = len(useful_terms & window_terms)

        if overlap:
            ranked.append((overlap, window))

    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1]["end"] - item[1]["start"],
            item[1]["start"],
        )
    )

    print("=" * 80)
    print("QUESTION:", question)
    print("QUERY TERMS:", sorted(useful_terms))

    for overlap, window in ranked[:5]:
        print(
            f"{window['start']:7.2f}-{window['end']:7.2f} "
            f"overlap={overlap} | {window['text']}"
        )

    print()
