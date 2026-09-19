"""Attempt 8, Step 3: coarse grid over v4 shrink params, nested 5-fold CV.

Params: ds, de, beta (F-beta), max_run, tau (fallback trigger),
shift_on_fallback. 720 combos. Fit = argmax mean-tIoU on 4 train folds,
test once on held-out fold. Success bar: pooled held-out delta >= +0.01
over the v4 baseline point. Also reports in-sample best (for the record).
"""

import itertools
import json
import sys

import numpy as np

from utils import gold_evidence, temporal_iou

from medical_evidence import normalize_tokens

sys.path.insert(0, "experiments")
from grid_params_v6 import BASELINE, COARSE_GRID

_STOP = set(
    "the a an is are was were be been do does did will would should can could "
    "have has had what when where who whom why how it its this that these "
    "those there here and or but of in on at to for with by from as".split()
)

import re

_SENT_END = re.compile(r"[.!?][\"')\]]*$")


def build_sentences(words):
    sentences, current = [], []
    for word in words:
        current.append(word)
        if _SENT_END.search(word["word"]):
            sentences.append(list(current))
            current = []
    if current:
        sentences.append(list(current))
    out = []
    for chunk in sentences:
        toks = normalize_tokens(" ".join(w["word"] for w in chunk))
        out.append(
            {
                "start": float(chunk[0]["start"]),
                "end": float(chunk[-1]["end"]),
                "tokens": toks,
            }
        )
    return out


def fbeta(inter, na, nb, beta):
    if inter <= 0 or na <= 0 or nb <= 0:
        return 0.0
    p = inter / nb
    r = inter / na
    b2 = beta * beta
    denom = b2 * p + r
    if denom <= 0:
        return 0.0
    return (1 + b2) * p * r / denom


def precompute(words, sentences, rows, qa):
    items = []
    for qi, r in enumerate(rows):
        ans, ids = qa[qi]
        if not (bool(ans) and int(r["label"]) == 1 and gold_evidence(r) is not None):
            items.append(None)
            continue
        n = len(sentences)
        ok = bool(ids) and all(
            isinstance(s, int) and not isinstance(s, bool) and 0 <= s < n for s in ids
        )
        if not ok:
            items.append({"gold": gold_evidence(r), "dead": True})
            continue
        qt = set(t for t in normalize_tokens(r["question"]) if t not in _STOP)
        cited = sorted(set(ids))
        cands = []
        for a in range(len(cited)):
            toks = []
            for b in range(a, len(cited)):
                if cited[b] != cited[a] + (b - a):
                    break
                s = sentences[cited[b]]
                toks.extend(s["tokens"])
                sb = set(t for t in toks if t not in _STOP)
                cands.append(
                    {
                        "s": sentences[cited[a]]["start"],
                        "e": s["end"],
                        "k": b - a + 1,
                        "inter": len(qt & sb),
                        "na": len(qt),
                        "nb": len(sb),
                    }
                )
        lo, hi = min(ids), max(ids)
        full = (sentences[lo]["start"], sentences[hi]["end"])
        items.append({"cands": cands, "full": full, "gold": gold_evidence(r)})
    return items


def span_for(item, ds, de, beta, max_run, tau, shift_fb):
    if item is None:
        return None
    if item.get("dead"):
        return None
    best, best_score = None, -1.0
    for c in item["cands"]:
        if c["k"] > max_run:
            continue
        sc = fbeta(c["inter"], c["na"], c["nb"], beta)
        if sc > best_score:
            best_score = sc
            best = c
    if best is None or best_score <= tau:
        s, e = item["full"]
        if shift_fb:
            s += ds
            e -= de
        return (s, e) if e > s and s >= 0 else item["full"]
    s = best["s"] + ds
    e = best["e"] - de
    if e <= s or s < 0:
        f = item["full"]
        if shift_fb:
            f = (f[0] + ds, f[1] - de)
            if f[1] <= f[0] or f[0] < 0:
                return item["full"]
        return f
    return (s, e)


def mean_tiou(items, params):
    ds, de, beta, max_run, tau, shift_fb = params
    vals = []
    for it in items:
        if it is None:
            continue
        if it.get("dead"):
            vals.append(0.0)
            continue
        vals.append(temporal_iou(it["gold"], span_for(it, ds, de, beta, max_run, tau, shift_fb)))
    return float(np.mean(vals)) if vals else 0.0


def main():
    cache = json.load(open("experiments/cache_v6.json"))
    splits = json.load(open("experiments/splits.json"))["folds"]

    convs = {}
    for audio_filename, info in cache.items():
        words = info["words"]
        sentences = build_sentences(words)
        convs[audio_filename] = precompute(words, sentences, info["rows"], info["qa"])

    combos = list(
        itertools.product(
            COARSE_GRID["ds"],
            COARSE_GRID["de"],
            COARSE_GRID["beta"],
            COARSE_GRID["max_run"],
            COARSE_GRID["tau"],
            COARSE_GRID["shift_on_fallback"],
        )
    )
    print(f"combos: {len(combos)}")
    base_p = (
        BASELINE["ds"],
        BASELINE["de"],
        BASELINE["beta"],
        BASELINE["max_run"],
        BASELINE["tau"],
        BASELINE["shift_on_fallback"],
    )

    # Baseline per fold + overall (faithfulness).
    fold_names = sorted(splits.keys())
    for f in fold_names:
        items = [it for cid in splits[f] for it in convs[cid]]
        print(f"{f}: v4-baseline tIoU={mean_tiou(items, base_p):.4f} (n={len(items)})")

    # In-sample best (whole data) — for the record only.
    all_items = [it for c in convs.values() for it in c]
    base_all = mean_tiou(all_items, base_p)
    print(f"v4-baseline overall: {base_all:.4f}")
    ib, ip = base_all, base_p
    for p in combos:
        v = mean_tiou(all_items, p)
        if v > ib:
            ib, ip = v, p
    print(f"IN-SAMPLE best: {ip} -> {ib:.4f} (delta {ib-base_all:+.4f})")

    # Nested CV: fit on 4 folds, test on held-out.
    results = {}
    for held in fold_names:
        train_items = [it for f in fold_names if f != held for cid in splits[f] for it in convs[cid]]
        bb, bp = -1.0, None
        for p in combos:
            v = mean_tiou(train_items, p)
            if v > bb:
                bb, bp = v, p
        test_items = [it for cid in splits[held] for it in convs[cid]]
        hb = mean_tiou(test_items, base_p)
        he = mean_tiou(test_items, bp)
        results[held] = {"params": bp, "base": hb, "exp": he, "n": len(test_items)}
        print(
            f"{held}: params={bp} base={hb:.4f} exp={he:.4f} "
            f"delta={he-hb:+.4f} (n={len(test_items)})"
        )

    hb_all = float(np.mean([results[f]["base"] for f in fold_names]))
    he_all = float(np.mean([results[f]["exp"] for f in fold_names]))
    improved = sum(1 for f in fold_names if results[f]["exp"] > results[f]["base"])

    # Bootstrap over conversations on held-out predictions.
    np.random.seed(42)
    conv_delta = {}
    for held in fold_names:
        bp = results[held]["params"]
        for cid in splits[held]:
            db, de_ = [], []
            for it in convs[cid]:
                if it is None:
                    continue
                if it.get("dead"):
                    db.append(0.0)
                    de_.append(0.0)
                    continue
                db.append(temporal_iou(it["gold"], span_for(it, *base_p)))
                de_.append(temporal_iou(it["gold"], span_for(it, *bp)))
            if db:
                conv_delta[cid] = float(np.mean(de_) - np.mean(db))
    cids = sorted(conv_delta.keys())
    boots = [
        float(np.mean([conv_delta[c] for c in np.random.choice(cids, size=len(cids), replace=True)]))
        for _ in range(1000)
    ]
    boot_rate = float(np.mean([d > 0 for d in boots]))

    print("=== NESTED-CV GATE (coarse) ===")
    print(f"held-out base {hb_all:.4f} -> exp {he_all:.4f} delta {he_all-hb_all:+.4f} (need >= +0.01)")
    print(f"improved folds: {improved}/5")
    print(f"bootstrap win rate: {boot_rate:.1%}")
    print(f"GATE: {'PASSED' if (he_all-hb_all) >= 0.01 else 'REJECTED'}")


if __name__ == "__main__":
    main()
