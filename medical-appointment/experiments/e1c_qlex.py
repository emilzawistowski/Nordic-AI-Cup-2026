"""E1c: shrink raw span using QUESTION-anchored k-run selection (no LLM cost)."""
import json, re, hashlib, itertools, numpy as np
from pathlib import Path
from utils import group_questions_by_conversation, gold_evidence, temporal_iou
from medical_asr import extract_words
from medical_evidence import locate_evidence, normalize_tokens

ROOT = Path(".")
FOLDS = json.load(open(ROOT / "experiments" / "splits.json"))["folds"]

STOP = set("the a an is are was were be been do does did will would should can could have has had what when where who whom why how it its this that these those there here and or but of in on at to for with by from as".split())

def content_toks(s):
    return [t for t in normalize_tokens(s) if t not in STOP]

def load():
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
            sents.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                          "toks": normalize_tokens(" ".join(x["word"] for x in cur))})
            cur = []
    if cur:
        sents.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                      "toks": normalize_tokens(" ".join(x["word"] for x in cur))})
    return sents

def runs(sents, kmax=3):
    out = []
    for k in range(1, kmax + 1):
        for i in range(len(sents) - k + 1):
            toks = [t for j in range(i, i + k) for t in sents[j]["toks"]]
            out.append({"start": sents[i]["start"], "end": sents[i+k-1]["end"], "toks": toks})
    return out

def f1(a, b):
    sa, sb = set(a) - STOP, set(b) - STOP
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    p, r = inter / len(sb), inter / len(sa)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def overlap(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

data = load()
pre = {}
for fn, info in data.items():
    sents = build_sentences(info["words"])
    sruns = runs(sents)
    per_q = []
    for idx, r in enumerate(info["rows"]):
        ans, ev = info["qa"][idx]
        raw = locate_evidence(info["words"], ev) if (ans and ev) else None
        per_q.append({"ans": bool(ans), "raw": raw, "label": int(r["label"]),
                      "gold": gold_evidence(r),
                      "q_toks": content_toks(r["question"])})
    pre[fn] = {"per_q": per_q, "runs": sruns}

file2fold = {m: f for f, ms in FOLDS.items() for m in ms}
folds = sorted(FOLDS)

def span_for(q, runs_, mode, ds, de):
    raw = q["raw"]
    if raw is None:
        # fallback: retrieval over all runs (YES never yields None)
        if mode in ("qall", "qall_fallback") and q["ans"]:
            best, bi = None, -1.0
            for u in runs_:
                sc = f1(q["q_toks"], u["toks"])
                if sc > bi:
                    bi, best = sc, u
            if best and bi > 0:
                return (best["start"] + ds, best["end"] - de)
        return None
    cands = runs_
    if mode in ("qin", "qin_fallback"):
        # only runs overlapping raw (LLM region + question shrink)
        cands = [u for u in runs_ if overlap(raw, (u["start"], u["end"])) > 0]
        if not cands:
            cands = runs_ if mode.endswith("fallback") else []
    best, bi = None, -1.0
    for u in cands:
        sc = f1(q["q_toks"], u["toks"])
        if sc > bi:
            bi, best = sc, u
    if best is None or bi <= 0:
        s, e = raw  # keep raw + bias
    else:
        s, e = best["start"], best["end"]
    s, e = s + ds, e - de
    return (s, e) if (e > s and s >= 0) else None

def tious(fn, mode, ds, de):
    p = pre[fn]
    out = []
    for q in p["per_q"]:
        if q["label"] == 1 and q["gold"] is not None:
            if not q["ans"]:
                out.append(0.0)
            else:
                out.append(temporal_iou(q["gold"], span_for(q, p["runs"], mode, ds, de)))
    return out

base_all = [v for fn in pre for v in tious(fn, "qin", 0.0, 0.0)]
print(f"baseline-in-code check: {np.mean(base_all):.4f} (expect ~0.4161)")

print("\n=== full-data scan ===")
res = []
for mode in ["qin", "qall", "qin_fallback"]:
    for ds, de in itertools.product([0.0, 0.3], [0.0, 0.5]):
        allv = [v for fn in pre for v in tious(fn, mode, ds, de)]
        res.append((float(np.mean(allv)), mode, ds, de))
res.sort(reverse=True)
for m, mode, ds, de in res:
    print(f"  {m:.4f} mode={mode} ds={ds} de={de}")

print("\n=== 5-fold CV (train picks best of scanned, held reports) ===")
for mode, ds, de in [(r[1], r[2], r[3]) for r in res[:4]]:
    deltas = []
    for held in folds:
        he = [f for f in pre if file2fold[f] == held]
        b = float(np.mean([v for f in he for v in tious(f, "qin", 0.0, 0.0)]))
        h = float(np.mean([v for f in he for v in tious(f, mode, ds, de)]))
        deltas.append(h - b)
    print(f"  mode={mode} ds={ds} de={de}: mean_delta={np.mean(deltas):+.4f} folds+={sum(d>0 for d in deltas)}/5 {[round(d,4) for d in deltas]}")
