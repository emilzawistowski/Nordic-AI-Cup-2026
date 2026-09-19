import json
import re
import numpy as np
from pathlib import Path
from utils import group_questions_by_conversation, gold_evidence, temporal_iou
from medical_evidence import locate_evidence
from medical_asr import extract_words

def load_cached_transcripts_and_llm():
    TRANSCRIPTS_DIR = Path("transcripts")
    LLM_CACHE_DIR = Path("experiments/llm_cache")
    
    conversations = group_questions_by_conversation()
    data = {}
    
    for audio_filename, rows in conversations:
        conv_id = audio_filename.replace("conversation_", "").replace(".mp3", "")
        t_file = TRANSCRIPTS_DIR / f"{conv_id}.json"
        
        if not t_file.exists():
            continue
            
        with open(t_file) as f:
            transcription = json.load(f)
            
        transcript_text = transcription.get("text", "").strip()
        words = extract_words(transcription)
        
        # Find cached LLM output
        questions = [r["question"] for r in rows]
        import hashlib
        q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        prompt_key = hashlib.md5(f"{transcript_text}\n{q_str}".encode()).hexdigest()
        llm_file = LLM_CACHE_DIR / f"{prompt_key}.json"
        
        qa_results = None
        if llm_file.exists():
            with open(llm_file) as f:
                qa_results = json.load(f)
                
        data[audio_filename] = {
            "transcription": transcription,
            "transcript_text": transcript_text,
            "words": words,
            "rows": rows,
            "qa_results": qa_results
        }
    return data

def run_diagnosis():
    data = load_cached_transcripts_and_llm()
    print(f"Loaded {len(data)} cached conversations.")
    
    gold_durations = []
    pred_durations = []
    tious = []
    
    bucket_counts = {"a_none_or_no": 0, "b_disjoint": 0, "c_inside": 0, "d_contains": 0, "e_partial": 0}
    bucket_loss = {"a_none_or_no": 0.0, "b_disjoint": 0.0, "c_inside": 0.0, "d_contains": 0.0, "e_partial": 0.0}
    
    start_diffs = []
    end_diffs = []
    
    # Boundary alignment data
    dist_whisper_seg_start = []
    dist_whisper_seg_end = []
    dist_pause_start = []
    dist_pause_end = []
    
    # Oracle ceilings
    best_whisper_seg_tious = []
    best_single_sentence_tious = []
    best_k_sentences_tious = []
    best_pause_turn_tious = []
    
    wrong_answers = {"false_yes": 0, "false_no": 0}
    yes_with_none_span = 0
    positives_per_conv = []
    
    total_positives = 0
    
    for audio_filename, item in data.items():
        rows = item["rows"]
        transcription = item["transcription"]
        words = item["words"]
        qa_results = item["qa_results"]
        
        segments = transcription.get("segments", [])
        
        # Split sentences from segments
        sentences = []
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            if not seg_text:
                continue
            # Simple sentence splitting by punctuation
            sents = re.split(r'(?<=[.!?])\s+', seg_text)
            # Rough time allocation per sentence by character count
            total_chars = sum(len(s) for s in sents) if sents else 1
            curr_t = seg_start
            for s in sents:
                dur = (len(s) / total_chars) * (seg_end - seg_start) if total_chars > 0 else 0
                sentences.append({"text": s, "start": curr_t, "end": curr_t + dur})
                curr_t += dur

        # Pause-based turns (pause > 0.25s)
        turns = []
        if words:
            curr_turn_start = words[0]["start"]
            curr_turn_words = [words[0]["word"]]
            for w1, w2 in zip(words[:-1], words[1:]):
                if w2["start"] - w1["end"] > 0.25:
                    turns.append({"start": curr_turn_start, "end": w1["end"], "text": " ".join(curr_turn_words)})
                    curr_turn_start = w2["start"]
                    curr_turn_words = [w2["word"]]
                else:
                    curr_turn_words.append(w2["word"])
            turns.append({"start": curr_turn_start, "end": words[-1]["end"], "text": " ".join(curr_turn_words)})

        pos_count = sum(1 for r in rows if int(r["label"]) == 1)
        positives_per_conv.append(pos_count)
        
        for idx, r in enumerate(rows):
            label = int(r["label"])
            gold = gold_evidence(r)
            
            ans, ev_text = qa_results[idx] if qa_results else (False, "")
            pred_ans = bool(ans)
            
            if pred_ans != bool(label):
                if pred_ans and label == 0:
                    wrong_answers["false_yes"] += 1
                elif not pred_ans and label == 1:
                    wrong_answers["false_no"] += 1

            if label == 1 and gold is not None:
                total_positives += 1
                g_start, g_end = gold
                gold_durations.append(g_end - g_start)
                
                # Boundary distance calculations
                # Whisper seg boundaries
                seg_starts = [s.get("start", 0.0) for s in segments]
                seg_ends = [s.get("end", 0.0) for s in segments]
                min_d_seg_start = min([abs(g_start - s) for s in seg_starts]) if seg_starts else 0
                min_d_seg_end = min([abs(g_end - s) for s in seg_ends]) if seg_ends else 0
                dist_whisper_seg_start.append(min_d_seg_start)
                dist_whisper_seg_end.append(min_d_seg_end)

                # Oracle ceilings
                # (i) Best Whisper segment
                best_seg_iou = max([temporal_iou(gold, (s["start"], s["end"])) for s in segments]) if segments else 0.0
                best_whisper_seg_tious.append(best_seg_iou)
                
                # (ii) Best single sentence
                best_sent_iou = max([temporal_iou(gold, (s["start"], s["end"])) for s in sentences]) if sentences else 0.0
                best_single_sentence_tious.append(best_sent_iou)
                
                # (iii) Best contiguous k sentences (k <= 3)
                best_k_iou = 0.0
                for k in range(1, 4):
                    for i in range(len(sentences) - k + 1):
                        span_k = (sentences[i]["start"], sentences[i + k - 1]["end"])
                        best_k_iou = max(best_k_iou, temporal_iou(gold, span_k))
                best_k_sentences_tious.append(best_k_iou)
                
                # (iv) Best pause turn
                best_turn_iou = max([temporal_iou(gold, (t["start"], t["end"])) for t in turns]) if turns else 0.0
                best_pause_turn_tious.append(best_turn_iou)

                # Predicted span
                pred_span = None
                if pred_ans and ev_text:
                    pred_span = locate_evidence(words, ev_text)
                
                if pred_ans and pred_span is None:
                    yes_with_none_span += 1

                iou = temporal_iou(gold, pred_span)
                tious.append(iou)
                loss = 1.0 - iou

                if pred_span is None or not pred_ans:
                    bucket_counts["a_none_or_no"] += 1
                    bucket_loss["a_none_or_no"] += loss
                else:
                    p_start, p_end = pred_span
                    pred_durations.append(p_end - p_start)
                    start_diffs.append(p_start - g_start)
                    end_diffs.append(p_end - g_end)
                    
                    if iou == 0.0:
                        bucket_counts["b_disjoint"] += 1
                        bucket_loss["b_disjoint"] += loss
                    elif p_start >= g_start and p_end <= g_end:
                        bucket_counts["c_inside"] += 1
                        bucket_loss["c_inside"] += loss
                    elif p_start <= g_start and p_end >= g_end:
                        bucket_counts["d_contains"] += 1
                        bucket_loss["d_contains"] += loss
                    else:
                        bucket_counts["e_partial"] += 1
                        bucket_loss["e_partial"] += loss

    print("\n--- PHASE 1 DIAGNOSTICS REPORT ---")
    print(f"Total Positives: {total_positives}")
    print(f"Mean tIoU baseline: {np.mean(tious):.4f}")
    
    print("\n1. DURATION DISTRIBUTION (seconds):")
    print(f"  Gold: Mean={np.mean(gold_durations):.2f}s, Median={np.median(gold_durations):.2f}s, P10={np.percentile(gold_durations, 10):.2f}s, P90={np.percentile(gold_durations, 90):.2f}s")
    if pred_durations:
        print(f"  Pred: Mean={np.mean(pred_durations):.2f}s, Median={np.median(pred_durations):.2f}s, P10={np.percentile(pred_durations, 10):.2f}s, P90={np.percentile(pred_durations, 90):.2f}s")

    print("\n2. LOSS DECOMPOSITION (out of 195 positives):")
    for b_key in ["a_none_or_no", "b_disjoint", "c_inside", "d_contains", "e_partial"]:
        cnt = bucket_counts[b_key]
        l_sum = bucket_loss[b_key]
        pct_loss = (l_sum / total_positives) if total_positives else 0
        print(f"  {b_key:<15}: count={cnt:3d} ({cnt/total_positives:5.1%}), total tIoU loss={l_sum:6.2f} (-{pct_loss:.4f} mean tIoU)")

    print("\n3. SYSTEMATIC BIAS (pred - gold):")
    if start_diffs:
        print(f"  Mean Start Diff: {np.mean(start_diffs):+.3f}s (Median: {np.median(start_diffs):+.3f}s)")
        print(f"  Mean End Diff:   {np.mean(end_diffs):+.3f}s (Median: {np.median(end_diffs):+.3f}s)")

    print("\n4. ORACLE CEILINGS (Best achievable mean tIoU):")
    print(f"  (i)   Best single Whisper segment: {np.mean(best_whisper_seg_tious):.4f}")
    print(f"  (ii)  Best single sentence:        {np.mean(best_single_sentence_tious):.4f}")
    print(f"  (iii) Best k contiguous sents (k<=3): {np.mean(best_k_sentences_tious):.4f}")
    print(f"  (iv)  Best pause turn (>0.25s):    {np.mean(best_pause_turn_tious):.4f}")

    print("\n5. ANSWER & SPAN QUALITY CHECK:")
    print(f"  Wrong answers: {wrong_answers}")
    print(f"  YES answers with None span: {yes_with_none_span}")
    print(f"  Positives per conv: min={min(positives_per_conv)}, max={max(positives_per_conv)}, all exact 5? {all(p == 5 for p in positives_per_conv)}")

if __name__ == "__main__":
    run_diagnosis()
