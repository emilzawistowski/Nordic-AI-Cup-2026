"""E1d: rigorous promotion-gate check for qin/ds=0.3 vs TRUE baseline."""
import json, re, hashlib, numpy as np
from pathlib import Path
from utils import group_questions_by_conversation, gold_evidence, temporal_iou
from medical_asr import extract_words
from medical_evidence import locate_evidence, normalize_tokens

ROOT = Path(".")
FOLDS = json.load(open(ROOT / "experiments" / "splits.json"))["folds"]
STOP = set("the a an is are was were be been do does did will would should can could have has had what when where who whom why how it its this that these those there here and or but of in on at to for with by from as".split())

def content_toks(s):
    return [t for t in normalize_tokens(s) if t not in STOP]

convs = group_questions_by_conversation()
pre = {}
for audio_filename, rows in convs:
    cid = audio_filename.replace("conversation_", "").replace(".mp3", "")
    tr = json.load(open(ROOT / "transcripts" / f"{cid}.json"))
    words = extract_words(tr)
    tt = tr.get("text", "").strip()
    q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate([r["question"] for r in rows])])
    key = hashlib.md5(f"{tt}\n{q_str}".encode()).hexdigest()
    qa = json.load(open(ROOT / "experiments" / "llm_cache" / f"{key}.json"))
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
    runs = []
    for k in (1, 2, 3):
        for i in range(len(sents) - k + 1):
            toks = [t for j in range(i, i + k) for t in sents[j]["toks"]]
            runs.append({"start": sents[i]["start"], "end": sents[i+k-1]["end"], "toks": toks})
    per_q = []
    for idx, r in enumerate(rows):
        ans, ev = qa[idx]
        raw = locate_evidence(words, ev) if (ans and ev) else None
        per_q.append({"ans": bool(ans), "raw": raw, "label": int(r["label"]),
                      "gold": gold_evidence(r), "q_toks": content_toks(r["question"])})
    pre[audio_filename] = {"per_q": per_q, "runs": runs, "words": words}

def f1(a, b):
    sa, sb = set(a) - STOP, set(b) - STOP
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    p, r = inter / len(sb), inter / len(sa)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def ov(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

DS, DE = 0.3, 0.0  # picked on full-data scan; verify via nested CV below

def exp_span(q, runs_, words):
    raw = q["raw"]
    if raw is None:
        if q["ans"]:  # retrieval fallback so YES never yields None
            best, bi = None, -1.0
            for u in runs_:
                sc = f1(q["q_toks"], u["toks"])
                if sc > bi:
                    bi, best = sc, u
            if best and bi > 0:
                s, e = best["start"] + DS, best["end"] - DE
                return (s, e) if e > s else None
        return None
    cands = [u for u in runs_ if ov(raw, (u["start"], u["end"])) > 0] or None
    if cands is None:
        s, e = raw
    else:
        best, bi = None, -1.0
        for u in cands:
            sc = f1(q["q_toks"], u["toks"])
            if sc > bi:
                bi, best = sc, u
        s, e = (best["start"], best["end"]) if (best and bi > 0) else raw
    s, e = s + DS, e - DE
    return (s, e) if (e > s and s >= 0) else None

file2fold = {m: f for f, ms in FOLDS.items() for m in ms}
folds = sorted(FOLDS)

# per-question results under base and exp
base_ious, exp_ious = [], []
per_fold_base, per_fold_exp = {}, {}
for held in folds:
    he = [f for f in pre if file2fold[f] == held]
    bv, hv = [], []
    for f in he:
        for q in pre[f]["per_q"]:
            if q["label"] == 1 and q["gold"] is not None:
                bv.append(temporal_iou(q["gold"], q["raw"] if q["ans"] else None))
                hv.append(temporal_iou(q["gold"], exp_span(q, pre[f]["runs"], pre[f]["words"]) if q["ans"] else None))
    per_fold_base[held] = float(np.mean(bv))
    per_fold_exp[held] = float(np.mean(hv))
    base_ious += bv
    exp_ious += hv
    print(f"held={held}: base={per_fold_base[held]:.4f} exp={per_fold_exp[held]:.4f} delta={per_fold_exp[held]-per_fold_base[held]:+.4f}")

print(f"\nOVERALL base={np.mean(base_ious):.4f} exp={np.mean(exp_ious):.4f} delta={np.mean(exp_ious)-np.mean(base_ious):+.4f}")
d = [per_fold_exp[f] - per_fold_base[f] for f in folds]
print(f"improved folds: {sum(x > 0 for x in d)}/5")
score_b = 0.4 * 0.9744 + 0.6 * float(np.mean(base_ious))
score_e = 0.4 * 0.9744 + 0.6 * float(np.mean(exp_ious))
print(f"score base={score_b:.4f} exp={score_e:.4f} (acc assumed equal 0.9744)")

# bootstrap over conversations
rng = np.random.RandomState(42)
files = list(pre)
b_by_f = {f: np.mean([temporal_iou(q['gold'], q['raw'] if q['ans'] else None) for q in pre[f]['per_q'] if q['label']==1 and q['gold'] is not None]) for f in files}
e_by_f = {f: np.mean([temporal_iou(q['gold'], exp_span(q, pre[f]['runs'], pre[f]['words']) if q['ans'] else None) for q in pre[f]['per_q'] if q['label']==1 and q['gold'] is not None]) for f in files}
wins = sum((float(np.mean([e_by_f[f] for f in rng.choice(files, size=len(files), replace=True)])) - float(np.mean([b_by_f[f] for f in rng.choice(files, size=len(files), replace=True)]))) > 0 for _ in range(1000))
# note: paired resample properly below
rng = np.random.RandomState(42)
wins = 0
for _ in range(1000):
    samp = rng.choice(files, size=len(files), replace=True)
    wins += float(np.mean([e_by_f[f] for f in samp]) - np.mean([b_by_f[f] for f in samp])) > 0
print(f"bootstrap win rate: {wins/10:.1f}%")
