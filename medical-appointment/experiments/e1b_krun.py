"""E1b: try k-sentence (1..3) selection + proportional shrink, with 5-fold CV."""
import json, re, hashlib, itertools, numpy as np
from pathlib import Path
from collections import Counter
from utils import group_questions_by_conversation, gold_evidence, temporal_iou
from medical_asr import extract_words
from medical_evidence import locate_evidence, normalize_tokens

ROOT = Path(".")
FOLDS = json.load(open(ROOT / "experiments" / "splits.json"))["folds"]

def load_cache():
    convs = group_questions_by_conversation()
    data = {}
    for audio_filename, rows in convs:
        cid = audio_filename.replace("conversation_", "").replace(".mp3", "")
        tr = json.load(open(ROOT / "transcripts" / f"{cid}.json"))
        words = extract_words(tr)
        tt = tr.get("text", "").strip()
        q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate([r["question"] for r in rows])])
        key = hashlib.md5(f"{tt}\n{q_str}".encode()).hexdigest()
        qa = json.load(open(ROOT / "experiments" / "llm_cache" / f"{key}.json"))
        data[audio_filename] = {"tr": tr, "words": words, "rows": rows, "qa": qa}
    return data

def build_sentences(words):
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?][\"')\]]*$", w["word"]):
            toks = normalize_tokens(" ".join(x["word"] for x in cur))
            sents.append({"start": cur[0]["start"], "end": cur[-1]["end"], "toks": toks})
            cur = []
    if cur:
        toks = normalize_tokens(" ".join(x["word"] for x in cur))
        sents.append({"start": cur[0]["start"], "end": cur[-1]["end"], "toks": toks})
    return sents

def runs(sents, kmax=3):
    out = []
    for k in range(1, kmax + 1):
        for i in range(len(sents) - k + 1):
            toks = [t for j in range(i, i + k) for t in sents[j]["toks"]]
            out.append({"start": sents[i]["start"], "end": sents[i+k-1]["end"], "toks": toks})
    return out

def iou(a, b):
    if a is None or b is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def f1(a_toks, b_toks):
    sa, sb = set(a_toks), set(b_toks)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    p = inter / len(sb)
    r = inter / len(sa)
    return 2 * p * r / (p + r) if (p + r) else 0.0

data = load_cache()
pre = {}
for fn, info in data.items():
    words, rows, qa = info["words"], info["rows"], info["qa"]
    sents = build_sentences(words)
    sruns = runs(sents, 3)
    per_q = []
    for idx, r in enumerate(rows):
        ans, ev = qa[idx]
        raw = locate_evidence(words, ev) if (ans and ev) else None
        per_q.append({"ans": bool(ans), "raw": raw, "label": int(r["label"]),
                      "gold": gold_evidence(r), "ev_toks": normalize_tokens(ev) if ev else []})
    pre[fn] = {"per_q": per_q, "sents": sents, "runs": sruns}

file2fold = {m: f for f, ms in FOLDS.items() for m in ms}
folds = sorted(FOLDS)

def predict_span(q, sents, runs_, mode, ds, de):
    raw = q["raw"]
    if raw is None:
        return None
    s, e = raw
    if mode in ("predrun", "lexrun"):
        best, bi = None, -1.0
        for u in runs_:
            sc = iou(raw, (u["start"], u["end"])) if mode == "predrun" else f1(q["ev_toks"], u["toks"])
            if sc > bi:
                bi, best = sc, u
        if best is not None and bi > 0:
            s, e = best["start"], best["end"]
        elif mode == "lexrun":
            return None  # no lexical match at all
    elif mode == "frac":
        c = (s + e) / 2
        half = (e - s) / 2 * ds  # ds reused as keep-fraction
        s, e = c - half, c + half
        return (s, e) if e > s else None
    s, e = s + ds, e - de
    return (s, e) if (e > s and s >= 0) else None

def file_tious(fn, mode, ds, de):
    p = pre[fn]
    out = []
    for q in p["per_q"]:
        if q["label"] == 1 and q["gold"] is not None:
            if not q["ans"]:
                out.append(0.0)
            else:
                out.append(temporal_iou(q["gold"], predict_span(q, p["sents"], p["runs"], mode, ds, de)))
    return out

# oracle check with exact sentences
for kmax, name in [(1, "1sent"), (3, "krun<=3")]:
    allv = []
    for fn, p in pre.items():
        for q in p["per_q"]:
            if q["label"] == 1 and q["gold"] is not None:
                allv.append(max([iou(q["gold"], (u["start"], u["end"])) for u in (runs(p["sents"], kmax))]))
    print(f"oracle {name}: {np.mean(allv):.4f}")

# full-data scan
configs = []
for mode in ["bias", "predrun", "lexrun"]:
    for ds, de in itertools.product([0.0, 0.3, 0.5], [0.0, 0.5, 1.0]):
        configs.append((mode, ds, de))
for frac in [0.3, 0.4, 0.5, 0.6]:
    configs.append(("frac", frac, 0.0))
print("\n=== full-data scan ===")
scored = []
for mode, ds, de in configs:
    allv = [v for fn in pre for v in file_tious(fn, mode, ds, de)]
    m = float(np.mean(allv))
    scored.append((m, mode, ds, de))
scored.sort(reverse=True)
for m, mode, ds, de in scored[:12]:
    print(f"  {m:.4f} mode={mode} ds={ds} de={de}")

# CV on top-6 distinct modes
cands = [(mode, ds, de) for _, mode, ds, de in scored[:6]]
print("\n=== 5-fold CV ===")
for mode, ds, de in cands:
    deltas = []
    for held in folds:
        tr = [f for f in pre if file2fold[f] != held]
        he = [f for f in pre if file2fold[f] == held]
        b = float(np.mean([v for f in he for v in file_tious(f, "bias", 0.0, 0.0)]))
        h = float(np.mean([v for f in he for v in file_tious(f, mode, ds, de)]))
        deltas.append(h - b)
    print(f"  mode={mode} ds={ds} de={de}: mean_delta={np.mean(deltas):+.4f} folds+={sum(d>0 for d in deltas)}/5 per-fold={[round(d,4) for d in deltas]}")
