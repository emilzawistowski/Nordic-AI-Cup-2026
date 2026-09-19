"""Attempt 9 shared loaders, folds, oracle-run search, scorer, bootstrap.

All offline steps use cached transcripts/ (no ASR re-runs).
Folds: spec fallback (grid_params_v6.py holds no split): sorted audio
filenames, random.Random(42).shuffle, fold = index % 5.
"""

import json
import random
from pathlib import Path

from medical_asr import extract_words
from utils import gold_evidence, temporal_iou

from medical_evidence_v4 import build_sentences

DATA_DIR = Path("transcripts")
START_SHIFT = 0.3


def load_conversation_words(audio_filename):
    conv_id = audio_filename.replace("conversation_", "").replace(".mp3", "")
    with open(DATA_DIR / f"{conv_id}.json") as f:
        transcription = json.load(f)
    return extract_words(transcription)


def conversation_ids():
    from utils import group_questions_by_conversation

    return sorted([fn for fn, _ in group_questions_by_conversation()])


def folds():
    ids = conversation_ids()
    shuffled = list(ids)
    random.Random(42).shuffle(shuffled)
    result = {f"fold_{k}": [] for k in range(5)}
    for i, cid in enumerate(shuffled):
        result[f"fold_{i % 5}"].append(cid)
    return result


def oracle_run(sentences, gold_span):
    """Best contiguous 1-3 sentence run under production geometry.

    Geometry: (S[i].start + 0.3, S[j].end); unshifted if end <= start + 0.05.
    Tie -> fewer sentences. Max tIoU == 0 -> single sentence nearest the
    gold midpoint. Returns (i, j, tiou).
    """
    gmin, gmax = gold_span
    gmid = (gmin + gmax) / 2.0
    best = None
    best_tiou = -1.0
    n = len(sentences)
    for i in range(n):
        for j in range(i, min(n, i + 3)):
            s = sentences[i]["start"] + START_SHIFT
            e = sentences[j]["end"]
            if e <= s + 0.05:
                s = sentences[i]["start"]
                e = sentences[j]["end"]
            t = temporal_iou(gold_span, (s, e))
            key = (t, -(j - i))
            if best is None or key > (best_tiou, -(best[1] - best[0])):
                best = (i, j)
                best_tiou = t
    if best_tiou <= 0:
        # Single sentence nearest in time to the gold midpoint.
        bi = min(
            range(n),
            key=lambda k: abs((sentences[k]["start"] + sentences[k]["end"]) / 2.0 - gmid),
        )
        s = sentences[bi]["start"] + START_SHIFT
        e = sentences[bi]["end"]
        if e <= s + 0.05:
            s = sentences[bi]["start"]
        best = (bi, bi)
        best_tiou = temporal_iou(gold_span, (s, e))
    return best[0], best[1], best_tiou


def score_lists(labels, golds, pred_answers, pred_spans):
    """Accuracy / mean tIoU / combined score from parallel lists."""
    tot = len(labels)
    cor = sum(1 for l, p in zip(labels, pred_answers) if bool(p) == bool(l))
    tious = [
        temporal_iou(g, s)
        for l, g, s in zip(labels, golds, pred_spans)
        if l == 1 and g is not None
    ]
    acc = cor / tot if tot else 0.0
    tiou = sum(tious) / len(tious) if tious else 0.0
    return {
        "accuracy": acc,
        "correct": cor,
        "total": tot,
        "mean_tiou": tiou,
        "n_tiou": len(tious),
        "score": 0.4 * acc + 0.6 * tiou,
    }


def paired_bootstrap(conv_scores_base, conv_scores_exp, seed=42, n=1000):
    """P(delta score > 0) over resampled conversations.

    Args: dicts conv_id -> score.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    cids = sorted(set(conv_scores_base) & set(conv_scores_exp))
    wins = 0
    for _ in range(n):
        samp = [cids[i] for i in rng.randint(0, len(cids), size=len(cids))]
        db = sum(conv_scores_base[c] for c in samp) / len(samp)
        de = sum(conv_scores_exp[c] for c in samp) / len(samp)
        if de - db > 0:
            wins += 1
    return wins / n
