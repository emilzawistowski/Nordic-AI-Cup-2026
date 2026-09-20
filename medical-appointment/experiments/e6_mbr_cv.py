"""Attempt 10 CV gate: MBR/majority-vote fusion over 5 sampled reasoner runs.

Phase 1 (MLX, resumable): for each conversation with a cached transcript in
transcripts/, run the existing indexed batched QA prompt 5x with temp=0.7,
top_p=0.9 and independent seeds (medical_reasoner_v6.VOTE_SEEDS) against the
SAME numbered transcript. Raw + parsed outputs cached per conversation in
experiments/e6_cache/ so the threshold sweep never re-runs inference.
Conversations with a missing cached transcript are SKIPPED (no ASR re-runs).

Phase 2 (offline): nested 5-fold CV reusing the EXACT attempt-9 fold split
(experiments/ft1_common.folds — imported, not regenerated). Per held-out
fold: sweep yes_threshold over {2, 3} on the 4 training folds (pooled
score = 0.4*acc + 0.6*tiou, tie -> 3/majority), evaluate the winner on the
held-out fold. Baseline = v4 deterministic QA from experiments/cache_v6.json
(re-run on the identical cached-transcript set, not HISTORY.md numbers).

Gate (per fold, exp - base): tIoU delta >= +0.02 AND accuracy delta >= 0.000
AND score delta > 0, in >= 4 of 5 folds.

Geometry/shrink (calibrate_indexed_span) is imported UNCHANGED via
medical_evidence_v6.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft1_common import folds, score_lists  # noqa: E402  (exact attempt-9 split)

from medical_asr import extract_words  # noqa: E402
from medical_evidence_v6 import (  # noqa: E402
    build_numbered_transcript,
    build_sentences,
    calibrate_indexed_span,
    content_tokens,
)
from medical_reasoner_v6 import (  # noqa: E402
    SAMPLE_TEMP,
    SAMPLE_TOP_P,
    VOTE_SEEDS,
    answer_questions_batch_indexed_sampled,
    fuse_votes,
)
from utils import (  # noqa: E402
    gold_evidence,
    group_questions_by_conversation,
    temporal_iou,
)

TRANSCRIPTS_DIR = Path("transcripts")
CACHE_DIR = Path("experiments/e6_cache")
BASELINE_CACHE = Path("experiments/cache_v6.json")
RESULTS_OUT = Path("experiments/e6_results.json")
THRESHOLDS = (2, 3)


def conv_id_of(audio_filename):
    return audio_filename.replace("conversation_", "").replace(".mp3", "")


def load_words(audio_filename):
    """Words from the cached Whisper transcript, or None if missing (skip)."""
    p = TRANSCRIPTS_DIR / f"{conv_id_of(audio_filename)}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return extract_words(json.load(f))


def build_cache(rows_by_conv):
    """Phase 1: 5x sampled LLM outputs per conversation (resumable)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    convs = sorted(rows_by_conv)
    missing = [c for c in convs if (CACHE_DIR / f"{conv_id_of(c)}.json").exists()]
    print(f"cache: {len(missing)}/{len(convs)} conversations already cached")
    t0 = time.time()
    done = 0
    for audio_filename in convs:
        out = CACHE_DIR / f"{conv_id_of(audio_filename)}.json"
        if out.exists():
            continue
        words = load_words(audio_filename)
        if words is None:
            print(f"  SKIP {audio_filename}: no cached transcript (no ASR re-run)")
            continue
        rows = rows_by_conv[audio_filename]
        sentences = build_sentences(words)
        numbered = build_numbered_transcript(sentences)
        questions = [r["question"] for r in rows]
        raw_list, parsed_list = [], []
        for seed in VOTE_SEEDS:
            raw, parsed = answer_questions_batch_indexed_sampled(
                numbered, questions,
                temp=SAMPLE_TEMP, top_p=SAMPLE_TOP_P, seed=seed,
            )
            raw_list.append(raw)
            parsed_list.append([[bool(a), list(ids)] for a, ids in parsed])
        out.write_text(json.dumps({
            "audio_filename": audio_filename,
            "words": words,
            "questions": questions,
            "seeds": list(VOTE_SEEDS),
            "temp": SAMPLE_TEMP,
            "top_p": SAMPLE_TOP_P,
            "raw": raw_list,
            "parsed": parsed_list,
        }))
        done += 1
        dt = time.time() - t0
        print(f"  [{done}] {audio_filename} cached ({dt:.0f}s elapsed)", flush=True)
    print(f"cache phase done: {done} new conversations in {time.time()-t0:.0f}s")


def load_samples(audio_filename):
    """Cached 5x parsed runs as [run][q] -> (bool, [ids]), or None."""
    p = CACHE_DIR / f"{conv_id_of(audio_filename)}.json"
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    return [[(bool(a), list(ids)) for a, ids in run] for run in info["parsed"]]


def score_convs(convs, rows_by_conv, get_qa):
    """Score a set of conversations; mirrors example.py (v4 geometry).

    get_qa(audio_filename, rows) -> list of (bool, [ids]) per question.
    Returns dict with accuracy/tiou/score + fallback/false-no counts.
    """
    labels, golds, answers, spans = [], [], [], []
    fallback, false_no = 0, 0
    for conv in convs:
        rows = rows_by_conv[conv]
        qa = get_qa(conv, rows)
        if conv in _words_cache:
            words = _words_cache[conv]
        else:
            words = load_words(conv)
            _words_cache[conv] = words
        sentences = build_sentences(words)
        for qi, r in enumerate(rows):
            label = int(r["label"])
            gold = gold_evidence(r)
            ans, ids = qa[qi]
            final = bool(ans)
            span = None
            if final:
                span, fell_back = calibrate_indexed_span(
                    sentences, ids, content_tokens(r["question"]))
                if span is None:
                    final = False
                elif fell_back:
                    fallback += 1
            if label == 1 and not final:
                false_no += 1
            labels.append(label)
            golds.append(gold)
            answers.append(final)
            spans.append(span)
    s = score_lists(labels, golds, answers, spans)
    s["shrink_fallback"] = fallback
    s["false_no"] = false_no
    return s


_words_cache = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--cv-only", action="store_true")
    args = ap.parse_args()

    rows_by_conv = {fn: rows for fn, rows in group_questions_by_conversation()}
    fold_map = folds()
    assert sorted(c for cs in fold_map.values() for c in cs) == sorted(rows_by_conv), \
        "fold split does not cover all conversations"
    print("fold split (attempt-9 ft1_common.folds): " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(fold_map.items())))

    if not args.cv_only:
        build_cache(rows_by_conv)
    if args.cache_only:
        return

    # --- Coverage: only conversations with BOTH e6 samples and v4 baseline ---
    base_cache = json.loads(BASELINE_CACHE.read_text())
    usable = [c for c in sorted(rows_by_conv)
              if load_samples(c) is not None and c in base_cache]
    skipped = sorted(set(rows_by_conv) - set(usable))
    if skipped:
        print(f"SKIPPED (no 5x samples or no baseline cache): {skipped}")
    print(f"scoring {len(usable)}/{len(rows_by_conv)} conversations")

    def base_qa(conv, rows):
        return [(bool(a), list(ids)) for a, ids in base_cache[conv]["qa"]]

    def exp_qa(threshold):
        def _get(conv, rows):
            return fuse_votes(load_samples(conv), yes_threshold=threshold)
        return _get

    fold_names = sorted(fold_map)
    results = {"thresholds": list(THRESHOLDS), "folds": {}}
    n_pass = 0
    for held in fold_names:
        held_convs = [c for c in fold_map[held] if c in usable]
        train_convs = [c for f in fold_names if f != held
                       for c in fold_map[f] if c in usable]
        # Sweep threshold on the 4 training folds (cached samples, no MLX).
        train_scores = {}
        for t in THRESHOLDS:
            train_scores[t] = score_convs(train_convs, rows_by_conv, exp_qa(t))["score"]
        best = max(THRESHOLDS, key=lambda t: (train_scores[t], t == 3))
        base = score_convs(held_convs, rows_by_conv, base_qa)
        exp = score_convs(held_convs, rows_by_conv, exp_qa(best))
        d_tiou = exp["mean_tiou"] - base["mean_tiou"]
        d_acc = exp["accuracy"] - base["accuracy"]
        d_score = exp["score"] - base["score"]
        passed = (d_tiou >= 0.02) and (d_acc >= 0.000) and (d_score > 0)
        n_pass += int(passed)
        results["folds"][held] = {
            "chosen_threshold": best,
            "train_scores": {str(t): train_scores[t] for t in THRESHOLDS},
            "baseline": base,
            "experiment": exp,
            "delta_tiou": d_tiou,
            "delta_acc": d_acc,
            "delta_score": d_score,
            "passed": passed,
        }
        print(
            f"{held}: thresh={best} (train " +
            " ".join(f"{t}:{train_scores[t]:.4f}" for t in THRESHOLDS) + ") | "
            f"base acc={base['accuracy']:.4f} tIoU={base['mean_tiou']:.4f} "
            f"score={base['score']:.4f} fb={base['shrink_fallback']} "
            f"fn={base['false_no']} | "
            f"exp acc={exp['accuracy']:.4f} tIoU={exp['mean_tiou']:.4f} "
            f"score={exp['score']:.4f} fb={exp['shrink_fallback']} "
            f"fn={exp['false_no']} | "
            f"d_tIoU={d_tiou:+.4f} d_acc={d_acc:+.4f} d_score={d_score:+.4f} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    # Pooled record (all usable convs, per-threshold + chosen-mix for reference).
    for t in THRESHOLDS:
        ov = score_convs(usable, rows_by_conv, exp_qa(t))
        print(f"pooled t={t}: acc={ov['accuracy']:.4f} tIoU={ov['mean_tiou']:.4f} "
              f"score={ov['score']:.4f} fb={ov['shrink_fallback']} fn={ov['false_no']}")
        results[f"pooled_t{t}"] = ov
    base_ov = score_convs(usable, rows_by_conv, base_qa)
    print(f"pooled BASE: acc={base_ov['accuracy']:.4f} tIoU={base_ov['mean_tiou']:.4f} "
          f"score={base_ov['score']:.4f} fb={base_ov['shrink_fallback']} "
          f"fn={base_ov['false_no']}")
    results["pooled_base"] = base_ov
    results["folds_passed"] = n_pass
    results["gate"] = "PASSED" if n_pass >= 4 else "REJECTED"
    print(f"=== GATE: {n_pass}/5 folds passed -> {results['gate']} "
          f"(need tIoU>=+0.02, acc>=+0.000, score>+0 in >=4/5 folds) ===")
    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_OUT}")


if __name__ == "__main__":
    main()
