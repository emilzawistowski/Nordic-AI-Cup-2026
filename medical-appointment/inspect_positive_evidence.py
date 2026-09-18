import csv
import json
from pathlib import Path


TRANSCRIPT_ID = "sample_4"
TRANSCRIPT_PATH = Path("transcripts/sample_4.json")
CSV_PATH = Path("data/question_train.csv")


result = json.loads(TRANSCRIPT_PATH.read_text(encoding="utf-8"))

words = [
    word
    for segment in result.get("segments", [])
    for word in segment.get("words", [])
]

with CSV_PATH.open(newline="", encoding="utf-8") as handle:
    rows = [
        row
        for row in csv.DictReader(handle)
        if row["transcript_id"] in {
            TRANSCRIPT_ID,
            f"conversation_{TRANSCRIPT_ID}",
        }
    ]

print("Questions found:", len(rows))
print("Words found:", len(words))
print()

for row in rows:
    if row["label"] != "1":
        continue

    evidence_start = float(row["evidence_start"])
    evidence_end = float(row["evidence_end"])

    evidence_words = [
        word["word"].strip()
        for word in words
        if float(word["end"]) >= evidence_start
        and float(word["start"]) <= evidence_end
    ]

    print("=" * 90)
    print("QUESTION:", row["question"])
    print("GOLD:", f"{evidence_start:.2f}-{evidence_end:.2f}")
    print("ASR:", " ".join(evidence_words))
    print()
