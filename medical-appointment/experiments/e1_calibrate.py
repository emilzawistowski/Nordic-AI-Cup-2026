"""E1 span calibration with proper 5-fold CV (fixes offline_eval conv-id bug).

Modes (<=4 discrete):
  raw             - baseline locate_evidence span unchanged
  bias            - raw shifted: start+DS, end-DE
  sent            - intersect raw span with best-overlapping sentence, then bias
  seg             - intersect raw span with best-overlapping whisper segment, then bias

Numeric params (<=2): DS (start shift), DE (end trim).
Grid: DS in {0.0, 0.3, 0.5}, DE in {0.0, 1.0, 1.5, 2.0}.

Sentences are built from WORD timings (split on tokens ending .?!),
so boundaries are exact, not char-proportioned.
"""
import json
import re
import hashlib
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

from utils import group_questions_by_conversation, gold_evidence, temporal_iou
from medical_asr import extract_words
from medical_evidence import locate_evidence

ROOT = Path(".")
SPLITS = json.load(open(ROOT / "experiments" / "splits.json"))
# splits.json folds already hold full filenames like "conversation_sample_64.mp3"
FOLDS = SPLITS["folds"]


def load_cache():
    convs = group_questions_by_conversation()
    data = {}
    for audio_filename, rows in convs:
        conv_id = audio_filename.replace("conversation_", "").replace(".mp3", "")
        tr = json.load(open(ROOT / "transcripts" / f"{conv_id}.json"))
        words = extract_words(tr)
        transcript_text = tr.get("text", "").strip()
        questions = [r["question"] for r in rows]
        q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        key = hashlib.md5(f"{transcript_text}\n{q_str}".encode()).hexdigest()
        qa = json.load(open(ROOT / "experiments" / "llm_cache" / f"{key}.json"))
        data[audio_filename] = {
            "transcription": tr, "words": words, "rows": rows,
            "qa": qa, "text": transcript_text,
        }
    return data


def build_sentences(words):
    """Split word list into sentences on tokens ending with . ? !"""
    sents = []
    cur = []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?][\"')\]]*$", w["word"]):
            sents.append({
                "start": cur[0]["start"], "end": cur[-1]["end"],
                "words": list(cur),
            })
            cur = []
    if cur:
        sents.append({"start": cur[0]["start"], "end": cur[-1]["end"], "words": cur})
    return sents


def raw_spans_for(data):
    """Precompute baseline (answer, raw_span) per question per conversation."""
    out = {}
    for audio_filename, info in data.items():
        words, rows, qa = info["words"], info["rows"], info["qa"]
        sents = build_sentences(words)
        segs = [(s.get("start", 0.0), s.get("end", 0.0))
                for s in info["transcription"].get("segments", [])]
        per_q = []
        for idx, r in enumerate(rows):
            ans, ev = qa[idx]
            span = None
            if ans and ev:
                span = locate_evidence(words, ev)
            per_q.append({"ans": bool(ans), "raw": span,
                          "label": int(r["label"]), "gold": gold_evidence(r)})
        out[audio_filename] = {"per_q": per_q, "sents": sents, "segs": segs}
    return out


def calibrate(raw, sents, segs, mode, ds, de):
    if raw is None:
        return None
    s, e = raw
    if mode in ("sent", "seg"):
        units = ([(x["start"], x["end"]) for x in sents] if mode == "sent" else segs)
        best, best_iou = None, -1.0
        for us, ue in units:
            inter = max(0.0, min(e, ue) - max(s, us))
            union = max(e, ue) - min(s, us)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best = iou, (us, ue)
        if best is not None and best_iou > 0:
            # intersect prediction with best unit (shrinks over-cover)
            s, e = max(s, best[0]), min(e, best[1])
            if e <= s:
                s, e = best
    if mode in ("bias", "sent", "seg"):
        s, e = s + ds, e - de
    if e <= s or s < 0:
        return None
    return (s, e)


def tiou_of_file(per_q, sents, segs, mode, ds, de):
    vals = []
    for q in per_q:
        if q["label"] == 1 and q["gold"] is not None:
            if not q["ans"]:
                vals.append(0.0)
            else:
                vals.append(temporal_iou(q["gold"], calibrate(q["raw"], sents, segs, mode, ds, de)))
    return vals


def main():
    data = load_cache()
    pre = raw_spans_for(data)
    # filename -> fold
    file2fold = {}
    for f, members in FOLDS.items():
        for m in members:
            file2fold[m] = f
    fold_names = sorted(FOLDS)

    DS_GRID = [0.0, 0.3, 0.5]
    DE_GRID = [0.0, 1.0, 1.5, 2.0]
    MODES = ["raw", "bias", "sent", "seg"]

    # full-data scores for reference
    print("=== Full-data reference ===")
    for mode in MODES:
        best = None
        for ds, de in itertools.product(DS_GRID, DE_GRID):
            if mode == "raw" and (ds, de) != (0.0, 0.0):
                continue
            allv = []
            for fn, p in pre.items():
                allv += tiou_of_file(p["per_q"], p["sents"], p["segs"], mode, ds, de)
            m = float(np.mean(allv))
            if best is None or m > best[0]:
                best = (m, ds, de)
        print(f"  {mode:5s}: best mean_tIoU={best[0]:.4f} @ ds={best[1]} de={best[2]}")

    # proper nested CV: for each held-out fold, pick params on other 4 folds
    print("\n=== 5-fold CV (params picked on train folds, scored on held-out) ===")
    base_held, cv_held = {}, {}
    chosen = {}
    for held in fold_names:
        train_files = [f for f in pre if file2fold.get(f) != held]
        held_files = [f for f in pre if file2fold.get(f) == held]
        # baseline on held
        bv = []
        for f in held_files:
            p = pre[f]
            bv += tiou_of_file(p["per_q"], p["sents"], p["segs"], "raw", 0.0, 0.0)
        base_held[held] = float(np.mean(bv)) if bv else 0.0
        # grid search on train
        best = None
        for mode in MODES:
            for ds, de in itertools.product(DS_GRID, DE_GRID):
                if mode == "raw" and (ds, de) != (0.0, 0.0):
                    continue
                tv = []
                for f in train_files:
                    p = pre[f]
                    tv += tiou_of_file(p["per_q"], p["sents"], p["segs"], mode, ds, de)
                m = float(np.mean(tv))
                if best is None or m > best[0]:
                    best = (m, mode, ds, de)
        _, mode, ds, de = best
        chosen[held] = (mode, ds, de)
        hv = []
        for f in held_files:
            p = pre[f]
            hv += tiou_of_file(p["per_q"], p["sents"], p["segs"], mode, ds, de)
        cv_held[held] = float(np.mean(hv)) if hv else 0.0
        print(f"  held={held}: train-pick mode={mode} ds={ds} de={de} "
              f"train_tIoU={best[0]:.4f} | held base={base_held[held]:.4f} held cv={cv_held[held]:.4f} "
              f"delta={cv_held[held]-base_held[held]:+.4f}")

    deltas = [cv_held[f] - base_held[f] for f in fold_names]
    print(f"\nMean held-out base={np.mean(list(base_held.values())):.4f} "
          f"cv={np.mean(list(cv_held.values())):.4f} "
          f"mean_delta={np.mean(deltas):+.4f}")
    print(f"Improved folds: {sum(d > 0 for d in deltas)}/5")

    # bootstrap over conversations for the CV-vs-base delta
    rng = np.random.RandomState(42)
    files = list(pre)
    # per-file mean tious under base and under chosen-per-fold params
    base_by_file, cv_by_file = {}, {}
    for f in files:
        p = pre[f]
        bv = tiou_of_file(p["per_q"], p["sents"], p["segs"], "raw", 0.0, 0.0)
        base_by_file[f] = float(np.mean(bv)) if bv else 0.0
        mode, ds, de = chosen[file2fold[f]]
        hv = tiou_of_file(p["per_q"], p["sents"], p["segs"], mode, ds, de)
        cv_by_file[f] = float(np.mean(hv)) if hv else 0.0
    wins = 0
    for _ in range(1000):
        samp = rng.choice(files, size=len(files), replace=True)
        d = float(np.mean([cv_by_file[f] for f in samp]) - np.mean([base_by_file[f] for f in samp]))
        wins += d > 0
    print(f"Bootstrap win rate: {wins/10:.1f}% (need >=90%)")
    # consensus params = most common choice
    from collections import Counter
    print("Chosen per fold:", chosen)
    print("Consensus:", Counter(chosen.values()).most_common(3))


if __name__ == "__main__":
    main()
