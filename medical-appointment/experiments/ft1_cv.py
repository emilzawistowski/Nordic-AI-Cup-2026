"""Attempt 9 CV driver: probe -> 5 fold runs (train + OOS eval) -> gate.

Resumable: skips training if models/medft1/{run}/adapters.safetensors exists;
skips eval where experiments/ft1_cache/{conv}.json exists.
Run under nohup; poll experiments/ft1_cv.log.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "experiments")
from ft1_common import (
    folds,
    load_conversation_words,
    paired_bootstrap,
    score_lists,
)

from utils import gold_evidence, group_questions_by_conversation, temporal_iou

CFG_DIR = Path("experiments/ft1_cfg")
ADAPTER_ROOT = Path("models/medft1")
CACHE_DIR = Path("experiments/ft1_cache")
LOG = Path("experiments/ft1_cv.log")

ITERS = {"fold0": 124, "fold1": 124, "fold2": 124, "fold3": 124, "fold4": 128,
         "full": 156}
LR = "2.0e-5"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wall_start():
    return int(open("experiments/ft1_run.log").read().strip().split()[0])


def elapsed():
    return time.time() - wall_start()


def maxlen():
    return int(open("experiments/ft1_maxlen.txt").read().strip())


def write_cfg(run, iters, adapter):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    ml = maxlen()
    cfg = f"""model: mlx-community/Qwen3-4B-Instruct-2507-4bit
train: true
fine_tune_type: lora
optimizer: adamw
data: experiments/ft1_data/{run}
adapter_path: {adapter}
seed: 42
num_layers: 16
batch_size: 1
iters: {iters}
val_batches: 1
steps_per_report: 5
steps_per_eval: 100000
save_every: {iters}
learning_rate: {LR}
max_seq_length: {ml}
mask_prompt: true
grad_checkpoint: false
lora_parameters:
  keys: [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj]
  rank: 16
  scale: 20.0
  dropout: 0.05
lr_schedule:
  name: cosine_decay
  warmup: 10
  warmup_init: 1.0e-7
  arguments: [{LR}, {iters}, 2.0e-6]
"""
    p = CFG_DIR / f"{run}.yaml"
    p.write_text(cfg)
    return p


def adapter_done(run):
    return (ADAPTER_ROOT / run / "adapters.safetensors").exists()


def run_lora(run, iters, adapter):
    cfg = write_cfg(run, iters, adapter)
    out = Path(f"experiments/ft1_log_{run}.txt")
    log(f"launch {run}: iters={iters} adapter={adapter} (log {out.name})")
    t0 = time.time()
    with open(out, "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "mlx_lm", "lora", "--config", str(cfg)],
            stdout=f, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"lora {run} failed, see {out}")
    log(f"done {run} in {dt:.0f}s")
    return out


def parse_s_it(logpath):
    times = []
    for line in open(logpath):
        m = re.search(r"Iter\s+(\d+).*?([\d.]+)\s*s/it", line, re.IGNORECASE)
        if m and 4 <= int(m.group(1)) <= 10:
            times.append(float(m.group(2)))
        else:
            m2 = re.search(r"Iter\s+(\d+).*?([\d.]+)\s*it/sec", line,
                           re.IGNORECASE)
            if m2 and 4 <= int(m2.group(1)) <= 10:
                times.append(1.0 / float(m2.group(2)))
    if not times:
        return None
    import numpy as np

    return float(np.mean(times))


def eval_fold(k, fold_convs, rows_by_conv):
    from mlx_lm import generate

    import medical_reasoner_ft1 as R
    from medical_reasoner_ft1 import build_messages
    from medical_reasoner_v2 import parse_indexed_response
    from medical_evidence_v4 import build_numbered_transcript, build_sentences

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = R.get_model(f"models/medft1/fold{k}")
    done = 0
    for conv in fold_convs:
        out = CACHE_DIR / f"{conv}.json"
        if out.exists():
            done += 1
            continue
        words = load_conversation_words(conv)
        sentences = build_sentences(words)
        numbered = build_numbered_transcript(sentences)
        questions = [r["question"] for r in rows_by_conv[conv]]
        prompt = tokenizer.apply_chat_template(
            build_messages(numbered, questions),
            add_generation_prompt=True, tokenize=False,
        )
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=600,
                       verbose=False)
        parsed = parse_indexed_response(raw, len(questions))
        out.write_text(json.dumps({"raw": raw, "parsed": parsed}))
        done += 1
        log(f"  eval fold{k}: {done}/{len(fold_convs)} {conv}")
    # Release the adapter model before the next stage.
    R._model_cache.pop(f"models/medft1/fold{k}", None)
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def bucket(gold, pred):
    if pred is None:
        return "none"
    gs, ge = gold
    ps, pe = pred
    if pe <= gs or ps >= ge:
        return "disjoint"
    if ps <= gs and pe >= ge:
        return "contains"
    if ps >= gs and pe <= ge:
        return "inside"
    return "partial"


def score_from_cache(rows_by_conv, fold_map, postproc, only_convs=None):
    """Score pooled OOS predictions for a post-processor ('A' or 'B').

    Returns (overall dict, per-fold dicts, per-conv scores, diagnostics).
    """
    from medical_evidence_ft1 import (
        build_sentences,
        calibrate_indexed_span,
        content_tokens,
        span_full_range_shifted,
    )

    labels, golds, answers, spans = [], [], [], []
    per_fold = {f: {"labels": [], "golds": [], "pred_answers": [], "pred_spans": []}
                for f in fold_map}
    per_conv = {}
    diag = {"unparsable": 0, "invalid_ids": 0, "false_no": 0, "false_yes": 0,
            "exact_match": 0, "n_pos": 0, "pred_len_hist": {},
            "oracle_len_hist": {}, "buckets": {}}
    sys.path.insert(0, "experiments")
    from ft1_common import oracle_run

    for conv, rows in rows_by_conv.items():
        if only_convs is not None and conv not in only_convs:
            continue
        words = load_conversation_words(conv)
        sentences = build_sentences(words)
        cached = json.loads((CACHE_DIR / f"{conv}.json").read_text())
        raw_lines = cached["raw"].strip().split("\n")
        parsed = [tuple(p) for p in cached["parsed"]]
        cla, cgo, can, csp = [], [], [], []
        for qi, r in enumerate(rows):
            label = int(r["label"])
            gold = gold_evidence(r)
            ans, ids = parsed[qi]
            ans = bool(ans)
            # Unparsable-line detection: no line matched this index.
            if not any(re.match(rf"^\s*{qi+1}\s*[\.\)]", ln.strip())
                       for ln in raw_lines):
                diag["unparsable"] += 1
            span = None
            final = ans
            if ans:
                valid = (
                    bool(ids) and all(
                        isinstance(s, int) and not isinstance(s, bool)
                        and 0 <= s < len(sentences) for s in ids
                    )
                )
                if not valid:
                    diag["invalid_ids"] += 1
                    final = False
                else:
                    if postproc == "A":
                        span = span_full_range_shifted(sentences, ids)
                    else:
                        qt = content_tokens(r["question"])
                        span, _ = calibrate_indexed_span(sentences, ids, qt)
                    if span is None:
                        final = False
            if label == 1 and not final:
                diag["false_no"] += 1
            if label == 0 and final:
                diag["false_yes"] += 1
            if label == 1 and gold is not None:
                diag["n_pos"] += 1
                if final and span is not None:
                    diag["buckets"][bucket(gold, span)] = (
                        diag["buckets"].get(bucket(gold, span), 0) + 1
                    )
                    a, b, _ = oracle_run(sentences, gold)
                    diag["oracle_len_hist"][b - a + 1] = (
                        diag["oracle_len_hist"].get(b - a + 1, 0) + 1
                    )
                    lo, hi = min(ids), max(ids)
                    plen = hi - lo + 1  # cited run length in sentences
                    diag["pred_len_hist"][plen] = (
                        diag["pred_len_hist"].get(plen, 0) + 1
                    )
                    if sorted(set(ids)) == list(range(a, b + 1)):
                        diag["exact_match"] += 1
            labels.append(label)
            golds.append(gold)
            answers.append(final)
            spans.append(span)
            cla.append(label)
            cgo.append(gold)
            can.append(final)
            csp.append(span)
        fold_of = next(f for f, cs in fold_map.items() if conv in cs)
        for key, vals in (("labels", cla), ("golds", cgo), ("pred_answers", can),
                          ("pred_spans", csp)):
            per_fold[fold_of][key].extend(vals)
        per_conv[conv] = score_lists(cla, cgo, can, csp)["score"]

    overall = score_lists(labels, golds, answers, spans)
    per_fold_scores = {f: score_lists(**per_fold[f]) for f in per_fold}
    return overall, per_fold_scores, per_conv, diag


def baseline_scores(rows_by_conv, fold_map, only_convs=None):
    cache = json.loads(open("experiments/cache_v6.json").read())
    from medical_evidence_v4 import (
        build_sentences,
        calibrate_indexed_span,
        content_tokens,
    )

    overall_l, overall_g, overall_a, overall_s = [], [], [], []
    per_fold = {f: {"labels": [], "golds": [], "pred_answers": [], "pred_spans": []}
                for f in fold_map}
    per_conv = {}
    for conv, rows in rows_by_conv.items():
        if only_convs is not None and conv not in only_convs:
            continue
        info = cache[conv]
        words = info["words"]
        sentences = build_sentences(words)
        cla, cgo, can, csp = [], [], [], []
        for qi, r in enumerate(rows):
            label = int(r["label"])
            gold = gold_evidence(r)
            ans, ids = info["qa"][qi]
            ans = bool(ans)
            span = None
            final = ans
            if ans:
                qt = content_tokens(r["question"])
                span, _ = calibrate_indexed_span(sentences, ids, qt)
                if span is None:
                    final = False
            for lst, v in ((overall_l, label), (overall_g, gold),
                           (overall_a, final), (overall_s, span)):
                lst.append(v)
            cla.append(label)
            cgo.append(gold)
            can.append(final)
            csp.append(span)
        fold_of = next(f for f, cs in fold_map.items() if conv in cs)
        for key, vals in (("labels", cla), ("golds", cgo), ("pred_answers", can),
                          ("pred_spans", csp)):
            per_fold[fold_of][key].extend(vals)
        per_conv[conv] = score_lists(cla, cgo, can, csp)["score"]
    overall = score_lists(overall_l, overall_g, overall_a, overall_s)
    return overall, {f: score_lists(**per_fold[f]) for f in per_fold}, per_conv


def main():
    rows_by_conv = {fn: rows for fn, rows in group_questions_by_conversation()}
    fold_map = folds()
    assert sorted([c for cs in fold_map.values() for c in cs]) == sorted(rows_by_conv)

    # --- Probe + budget gate ---
    if not adapter_done("_probe"):
        if elapsed() + 3600 > 14400:
            log("ABORT: no wall-clock room for probe")
            return
        t0 = time.time()
        out = run_lora("fold0", 10, "models/medft1/_probe")
        probe_wall = time.time() - t0
        s_it = parse_s_it(out) or probe_wall / 10
        Path("experiments/ft1_sit.txt").write_text(f"{s_it}")
    else:
        log("skip probe (adapter exists)")
        try:
            s_it = float(Path("experiments/ft1_sit.txt").read_text().strip())
        except Exception:
            s_it = parse_s_it(Path("experiments/ft1_log_fold0.txt"))
    total_iters = sum(ITERS[k] for k in
                      ["fold0", "fold1", "fold2", "fold3", "fold4", "full"])
    projected = s_it * total_iters + 600
    log(f"probe s_it={s_it:.2f}s total_iters={total_iters} "
        f"projected_total={projected:.0f}s (budget 12600s)")
    if projected > 12600:
        log("ABORT: projected total exceeds 3.5h budget")
        return

    # --- Fold runs ---
    for k in range(5):
        run = f"fold{k}"
        fkey = f"fold_{k}"
        if not adapter_done(run):
            need = s_it * ITERS[run] + 1200
            if elapsed() + need > 14400:
                log(f"ABORT: no wall-clock room for {run}")
                return
            run_lora(run, ITERS[run], f"models/medft1/{run}")
        else:
            log(f"skip training {run} (adapter exists)")
        eval_fold(k, fold_map[fkey], rows_by_conv)

        if k == 0:
            done_convs = set(fold_map[fkey])
            base, _, _ = baseline_scores(rows_by_conv, fold_map)
            for pp in ("A", "B"):
                ov, pf, _, _ = score_from_cache(
                    rows_by_conv, fold_map, pp, only_convs=done_convs)
                log(f"after fold0 [{pp}]: tIoU={ov['mean_tiou']:.4f} "
                    f"acc={ov['accuracy']:.4f}")
            # Early stop 1: parse health on fold0's evaluated convs.
            _, _, _, diag = score_from_cache(
                rows_by_conv, fold_map, "B", only_convs=done_convs)
            nq = len(fold_map[fkey]) * 10
            bad = diag["unparsable"] + diag["invalid_ids"]
            # diag is pooled over evaluated convs only (cache-gated).
            log(f"fold0 parse health: bad={bad} (unpars={diag['unparsable']} "
                f"invalid={diag['invalid_ids']})")
            if bad > 0.05 * nq:
                log("ABORT: >5% unparsable/invalid after fold 0")
                return

        if k == 1:
            # Early stop 2: pooled folds 0-1 tIoU delta < 0 for both A and B.
            f01 = set(fold_map["fold_0"]) | set(fold_map["fold_1"])
            b_ov, _, _ = baseline_scores(rows_by_conv, fold_map,
                                         only_convs=f01)
            early = {}
            for pp in ("A", "B"):
                ov, _, _, _ = score_from_cache(rows_by_conv, fold_map, pp,
                                               only_convs=f01)
                early[pp] = ov["mean_tiou"] - b_ov["mean_tiou"]
            log(f"folds 0-1 tIoU delta vs baseline: "
                f"A={early['A']:+.4f} B={early['B']:+.4f}")
            if early["A"] < 0 and early["B"] < 0:
                log("ABORT: folds 0-1 tIoU delta < 0 for both A and B")
                Path("experiments/ft1_results.json").write_text(json.dumps(
                    {"aborted": True, "stage": "after-fold1",
                     "delta_A": early["A"], "delta_B": early["B"],
                     "baseline_tiou": b_ov["mean_tiou"]}, indent=2))
                return

    # --- Full-pool gate ---
    base, _, _ = baseline_scores(rows_by_conv, fold_map)
    write_results(rows_by_conv, fold_map, base, aborted=False)


def write_results(rows_by_conv, fold_map, base, aborted):
    base_ov, base_pf, base_pc = base
    out = {"aborted": aborted,
           "baseline": {"overall": base_ov,
                        "per_fold": {f: base_pf[f] for f in base_pf}}}
    for pp in ("A", "B"):
        ov, pf, pc, diag = score_from_cache(rows_by_conv, fold_map, pp)
        dscore = ov["score"] - base_ov["score"]
        dtiou = ov["mean_tiou"] - base_ov["mean_tiou"]
        dacc = ov["accuracy"] - base_ov["accuracy"]
        g3 = sum(1 for f in pf if pf[f]["score"] > base_pf[f]["score"])
        g4 = paired_bootstrap(base_pc, pc)
        out[pp] = {"overall": ov, "per_fold": {f: pf[f] for f in pf},
                   "delta_tiou": dtiou, "delta_acc": dacc,
                   "delta_score": dscore, "folds_improved": g3,
                   "bootstrap": g4, "diagnostics": diag}
        log(f"[{pp}] pooled: acc={ov['accuracy']:.4f} (d{dacc:+.4f}) "
            f"tIoU={ov['mean_tiou']:.4f} (d{dtiou:+.4f}) "
            f"score={ov['score']:.4f} (d{dscore:+.4f}) "
            f"folds>{g3}/5 boot={g4:.3f}")
    Path("experiments/ft1_results.json").write_text(json.dumps(out, indent=2))
    log("wrote experiments/ft1_results.json")


if __name__ == "__main__":
    main()
