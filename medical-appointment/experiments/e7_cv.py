"""Attempt 11 CV: v4 baseline vs v7 cross-encoder fallback repair.

Single 5-fold evaluation (no fitting: cross-encoder is pretrained, threshold
is a fixed constant) on the EXACT attempt-9 split (ft1_common.folds).
Cached transcripts/words + cached call-1 LLM outputs from
experiments/cache_v6.json — only the cross-encoder runs fresh.

Gate: pooled d_tIoU >= +0.010 AND d_acc >= 0.000 AND per-fold d_tIoU >= 0 in
>= 4/5 folds AND repair fired on >= 15 of the baseline fallback cases.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft1_common import folds, score_lists  # noqa: E402 (exact attempt-9 split)

import medical_evidence_v7 as EV7  # noqa: E402
from medical_evidence_v4 import (  # noqa: E402
    build_sentences,
    calibrate_indexed_span as calibrate_v4,
    content_tokens,
)
from utils import (  # noqa: E402
    gold_evidence,
    group_questions_by_conversation,
    temporal_iou,
)

BASELINE_CACHE = Path("experiments/cache_v6.json")
RESULTS_OUT = Path("experiments/e7_results.json")


def main():
    t0 = time.time()
    rows_by_conv = {fn: rows for fn, rows in group_questions_by_conversation()}
    fold_map = folds()
    assert sorted(c for cs in fold_map.values() for c in cs) == sorted(rows_by_conv)
    print("fold split (attempt-9 ft1_common.folds): " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(fold_map.items())))
    base_cache = json.loads(BASELINE_CACHE.read_text())
    usable = [c for c in sorted(rows_by_conv) if c in base_cache]
    assert len(usable) == len(rows_by_conv), "baseline cache must cover all convs"

    fold_names = sorted(fold_map)
    results = {"folds": {}}
    all_b = {"labels": [], "golds": [], "answers": [], "spans": []}
    all_e = {"labels": [], "golds": [], "answers": [], "spans": []}
    total_fired = 0
    total_triggers = 0
    repaired_better = 0
    repaired_worse = 0
    all_log = []

    for held in fold_names:
        EV7.REPAIR_LOG.clear()
        held_convs = [c for c in fold_map[held] if c in usable]
        bl, bg, ba, bs = [], [], [], []
        el, eg, ea, es = [], [], [], []
        fold_fired = fold_better = fold_worse = 0
        for conv in held_convs:
            rows = rows_by_conv[conv]
            words = base_cache[conv]["words"]
            sentences = build_sentences(words)
            qa = base_cache[conv]["qa"]
            for qi, r in enumerate(rows):
                label = int(r["label"])
                gold = gold_evidence(r)
                ans, ids = qa[qi]
                final_b = final_e = bool(ans)
                span_b = span_e = None
                fb_b = fb_e = False
                if final_b:
                    qt = content_tokens(r["question"])
                    span_b, fb_b = calibrate_v4(sentences, ids, qt)
                    span_e, fb_e = EV7.calibrate_indexed_span(
                        sentences, ids, qt, question_text=r["question"])
                    if span_b is None:
                        final_b = False
                    if span_e is None:
                        final_e = False
                if label == 1 and not final_b:
                    assert not final_e, f"{conv} q{qi}: answer side changed!"
                bl.append(label); bg.append(gold); ba.append(final_b); bs.append(span_b)
                el.append(label); eg.append(gold); ea.append(final_e); es.append(span_e)
                if fb_b and span_b is not None and label == 1 and gold is not None:
                    tb = temporal_iou(gold, span_b)
                    te = temporal_iou(gold, span_e)
                    if not fb_e:  # repair fired and held
                        fold_fired += 1
                        if te > tb:
                            fold_better += 1
                        elif te < tb:
                            fold_worse += 1
        assert ba == ea, f"{held}: answers differ (must be exactly 0 delta)"
        sb = score_lists(bl, bg, ba, bs)
        se = score_lists(el, eg, ea, es)
        log = list(EV7.REPAIR_LOG)
        n_trig = len(log)
        n_fired = sum(1 for e in log if e["fired"])
        assert n_fired == fold_fired, f"{held}: log/fb mismatch {n_fired}/{fold_fired}"
        d_tiou = se["mean_tiou"] - sb["mean_tiou"]
        results["folds"][held] = {
            "baseline": sb, "experiment": se, "delta_tiou": d_tiou,
            "delta_acc": se["accuracy"] - sb["accuracy"],
            "delta_score": se["score"] - sb["score"],
            "triggers": n_trig, "fired": n_fired,
            "repaired_better": fold_better, "repaired_worse": fold_worse,
        }
        for k, v in (("labels", bl), ("golds", bg), ("answers", ba), ("spans", bs)):
            all_b[k].extend(v)
        for k, v in (("labels", el), ("golds", eg), ("answers", ea), ("spans", es)):
            all_e[k].extend(v)
        total_fired += n_fired
        total_triggers += n_trig
        repaired_better += fold_better
        repaired_worse += fold_worse
        all_log.extend(log)
        print(f"{held}: base tIoU={sb['mean_tiou']:.4f} acc={sb['accuracy']:.4f} "
              f"-> exp tIoU={se['mean_tiou']:.4f} acc={se['accuracy']:.4f} "
              f"d_tIoU={d_tiou:+.4f} triggers={n_trig} fired={n_fired} "
              f"better={fold_better} worse={fold_worse}", flush=True)

    # Baseline fallback count (score_lists has no fb field; recount directly).
    base_fb_total = 0
    for conv in usable:
        rows = rows_by_conv[conv]
        sentences = build_sentences(base_cache[conv]["words"])
        for qi, r in enumerate(rows):
            ans, ids = base_cache[conv]["qa"][qi]
            if ans:
                _, fb = calibrate_v4(
                    sentences, ids, content_tokens(r["question"]))
                base_fb_total += int(bool(fb))

    pooled_b = score_lists(all_b["labels"], all_b["golds"], all_b["answers"], all_b["spans"])
    pooled_e = score_lists(all_e["labels"], all_e["golds"], all_e["answers"], all_e["spans"])
    pd_tiou = pooled_e["mean_tiou"] - pooled_b["mean_tiou"]
    pd_acc = pooled_e["accuracy"] - pooled_b["accuracy"]
    pd_score = pooled_e["score"] - pooled_b["score"]
    folds_nonneg = sum(1 for f in fold_names
                       if results["folds"][f]["delta_tiou"] >= 0)
    ce_scores = [e["ce_score"] for e in all_log if e["ce_score"] is not None]
    run_lens = [e["run_len"] for e in all_log if e["fired"]]
    print(f"pooled BASE: acc={pooled_b['accuracy']:.4f} tIoU={pooled_b['mean_tiou']:.4f} "
          f"score={pooled_b['score']:.4f}")
    print(f"pooled EXP : acc={pooled_e['accuracy']:.4f} tIoU={pooled_e['mean_tiou']:.4f} "
          f"score={pooled_e['score']:.4f}")
    print(f"d_tIoU={pd_tiou:+.4f} (need >=+0.010) d_acc={pd_acc:+.4f} (need >=0) "
          f"d_score={pd_score:+.4f}")
    print(f"folds non-negative: {folds_nonneg}/5 (need >=4)")
    print(f"baseline fallbacks: {base_fb_total}; repair triggers: {total_triggers}; "
          f"fired: {total_fired} (need >=15); repaired better/worse: "
          f"{repaired_better}/{repaired_worse}")
    print(f"CE scores on triggers: min={min(ce_scores):.2f} "
          f"max={max(ce_scores):.2f} (threshold {EV7.CE_MIN_SCORE})" if ce_scores else "no CE scores")
    print(f"repaired run lengths: {sorted(run_lens)}")

    gate = (pd_tiou >= 0.010 and pd_acc >= 0.000 and folds_nonneg >= 4
            and total_fired >= 15)
    results.update({
        "pooled_base": pooled_b, "pooled_exp": pooled_e,
        "pooled_delta_tiou": pd_tiou, "pooled_delta_acc": pd_acc,
        "pooled_delta_score": pd_score, "folds_nonneg": folds_nonneg,
        "baseline_fallbacks": base_fb_total, "triggers": total_triggers,
        "fired": total_fired, "repaired_better": repaired_better,
        "repaired_worse": repaired_worse,
        "ce_model": EV7.CE_MODEL_NAME, "ce_min_score": EV7.CE_MIN_SCORE,
        "gate": "PASSED" if gate else "REJECTED",
    })
    print(f"=== GATE: {results['gate']} ({time.time()-t0:.0f}s wall) ===")
    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_OUT}")


if __name__ == "__main__":
    main()
