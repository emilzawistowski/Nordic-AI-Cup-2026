"""Attempt 9 data builder: oracle targets + P=4 variants per conversation.

Usage: PYTHONPATH=. .venv/bin/python experiments/ft1_build_data.py [all|foldK|full]
Builds experiments/ft1_data/{run}/train.jsonl + train_index.json + valid.jsonl.
Runs the oracle SANITY GATE (mean oracle tIoU over 195 positives in
[0.74, 0.83]), the prompt-parity assert, and the MAXLEN check.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, "experiments")
from ft1_common import folds, load_conversation_words, oracle_run

from medical_evidence_v4 import build_numbered_transcript, build_sentences
from medical_reasoner_ft1 import build_messages
from utils import gold_evidence, group_questions_by_conversation

P = 4
DATA_ROOT = Path("experiments/ft1_data")


def load_rows():
    return {
        fn: rows for fn, rows in group_questions_by_conversation()
    }


def assistant_line(i, is_pos, run):
    if not is_pos:
        return f"{i}. NO"
    a, b = run
    if a == b:
        return f"{i}. YES [{a}]"
    return f"{i}. YES [{a},{b}]"


def build_run(run_name, train_convs, rows_by_conv):
    outdir = DATA_ROOT / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []
    index = []
    for conv in train_convs:
        rows = rows_by_conv[conv]
        words = load_conversation_words(conv)
        sentences = build_sentences(words)
        numbered = build_numbered_transcript(sentences)
        assert len(rows) == 10, f"{conv}: {len(rows)} rows"
        # Per-question target from gold.
        targets = []
        for r in rows:
            if int(r["label"]) == 1:
                a, b, _ = oracle_run(sentences, gold_evidence(r))
                targets.append((True, (a, b)))
            else:
                targets.append((False, None))
        for v in range(P):
            order = list(range(10))
            if v > 0:
                random.Random(f"{conv}|{v}").shuffle(order)
            q_ordered = [rows[i]["question"] for i in order]
            msgs = build_messages(numbered, q_ordered)
            ans_lines = []
            for new_i, old_i in enumerate(order, start=1):
                is_pos, run = targets[old_i]
                ans_lines.append(assistant_line(new_i, is_pos, run))
            msgs = msgs + [{"role": "assistant", "content": "\n".join(ans_lines)}]
            lines.append({"messages": msgs})
            index.append(conv)
    with open(outdir / "train.jsonl", "w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
    with open(outdir / "train_index.json", "w") as f:
        json.dump(index, f)
    with open(outdir / "valid.jsonl", "w") as f:
        for obj in lines[:2]:
            f.write(json.dumps(obj) + "\n")
    print(f"{run_name}: {len(lines)} train examples, valid=2")
    return lines


def oracle_sanity(rows_by_conv):
    tious = []
    hist = {1: 0, 2: 0, 3: 0}
    for conv, rows in rows_by_conv.items():
        words = load_conversation_words(conv)
        sentences = build_sentences(words)
        for r in rows:
            if int(r["label"]) == 1:
                a, b, t = oracle_run(sentences, gold_evidence(r))
                tious.append(t)
                hist[b - a + 1] += 1
    import numpy as np

    mean = float(np.mean(tious))
    below = sum(1 for t in tious if t < 0.3)
    print(f"ORACLE sanity: mean tIoU={mean:.4f} (need [0.74,0.83]), n={len(tious)}")
    print(f"  run-length hist (1/2/3): {hist[1]}/{hist[2]}/{hist[3]}")
    print(f"  targets with tIoU < 0.3: {below}")
    assert len(tious) == 195, f"expected 195 positives, got {len(tious)}"
    assert 0.74 <= mean <= 0.83, f"ORACLE SANITY GATE FAILED: {mean:.4f}"
    print("ORACLE SANITY GATE PASSED")


def prompt_parity_and_maxlen(all_lines):
    import medical_reasoner_v2 as v2

    captured = {}

    def fake_generate(model, tokenizer, prompt, **kwargs):
        captured["prompt"] = prompt
        return "1. NO"

    v2.generate = fake_generate
    from medical_reasoner_ft1 import get_model

    _, tokenizer = get_model(None)
    conv_rows = load_rows()
    sample_conv = sorted(conv_rows.keys())[0]
    words = load_conversation_words(sample_conv)
    sentences = build_sentences(words)
    numbered = build_numbered_transcript(sentences)
    questions = [r["question"] for r in conv_rows[sample_conv]]
    v2.answer_questions_batch_indexed(numbered, questions)
    expected = tokenizer.apply_chat_template(
        build_messages(numbered, questions),
        add_generation_prompt=True,
        tokenize=False,
    )
    assert captured["prompt"] == expected, "PROMPT PARITY FAILED"
    print("PROMPT PARITY PASSED (v2 render == ft1 build_messages render)")

    max_tokens = 0
    for obj in all_lines:
        toks = tokenizer.apply_chat_template(
            obj["messages"], add_generation_prompt=False, tokenize=True
        )
        max_tokens = max(max_tokens, len(toks))
    maxlen = math.ceil((max_tokens + 64) / 256) * 256
    print(f"MAXLEN: max_tokens={max_tokens} -> MAXLEN={maxlen}")
    assert maxlen <= 6144, f"MAXLEN {maxlen} > 6144, abort"
    with open("experiments/ft1_maxlen.txt", "w") as f:
        f.write(str(maxlen))
    return maxlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="all")
    args = ap.parse_args()

    rows_by_conv = load_rows()
    assert len(rows_by_conv) == 39

    oracle_sanity(rows_by_conv)

    f = folds()
    all_convs = sorted(rows_by_conv.keys())
    assert sorted([c for fold in f.values() for c in fold]) == all_convs
    assert len({c for fold in f.values() for c in fold}) == 39

    runs = {}
    for k in range(5):
        held = set(f[f"fold_{k}"])
        train = [c for c in all_convs if c not in held]
        assert not (set(train) & held), "train/held-out overlap!"
        runs[f"fold{k}"] = train
    runs["full"] = all_convs

    which = (
        [args.run]
        if args.run != "all"
        else ["fold0", "fold1", "fold2", "fold3", "fold4", "full"]
    )
    all_lines = []
    for r in which:
        all_lines.extend(build_run(r, runs[r], rows_by_conv))

    prompt_parity_and_maxlen(all_lines)
    print("DATA BUILD COMPLETE")


if __name__ == "__main__":
    main()
