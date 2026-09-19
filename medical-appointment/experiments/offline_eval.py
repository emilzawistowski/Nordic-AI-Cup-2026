import json
import numpy as np
from pathlib import Path
from utils import group_questions_by_conversation, gold_evidence, temporal_iou, mean_temporal_iou

def load_splits():
    splits_file = Path("experiments/splits.json")
    if not splits_file.exists():
        raise FileNotFoundError("experiments/splits.json does not exist.")
    with open(splits_file) as f:
        return json.load(f)

def evaluate_pipeline_offline(predict_pipeline_fn, cached_data):
    """
    predict_pipeline_fn takes (audio_filename, transcript_info) and returns:
        answers: list of bools
        spans: list of (start, end) tuples or Nones
    """
    splits = load_splits()
    folds = splits["folds"]
    
    all_q_results = [] # list of dicts for each question
    fold_tious = {fold_name: [] for fold_name in folds}
    fold_accs = {fold_name: [] for fold_name in folds}
    
    # Map conv_id to audio_filename
    conversations = group_questions_by_conversation()
    conv_id_to_filename = {conv_id: filename for filename, rows in conversations for conv_id in [rows[0]['transcript_id']]}

    for fold_name, conv_ids in folds.items():
        for conv_id in conv_ids:
            audio_filename = f"conversation_{conv_id}.mp3"
            if audio_filename not in cached_data:
                continue
            
            info = cached_data[audio_filename]
            rows = info["rows"]
            answers, spans = predict_pipeline_fn(audio_filename, info)
            
            for idx, r in enumerate(rows):
                label = int(r["label"])
                gold = gold_evidence(r)
                pred_ans = bool(answers[idx])
                pred_span = spans[idx]
                
                is_correct = int(pred_ans == bool(label))
                iou = 0.0
                if label == 1 and gold is not None:
                    iou = temporal_iou(gold, pred_span)
                    fold_tious[fold_name].append(iou)
                
                fold_accs[fold_name].append(is_correct)
                all_q_results.append({
                    "fold": fold_name,
                    "conv_id": conv_id,
                    "q_id": r["question_id"],
                    "q_type": r["question_type"],
                    "label": label,
                    "gold_span": gold,
                    "pred_ans": pred_ans,
                    "pred_span": pred_span,
                    "is_correct": is_correct,
                    "iou": iou
                })

    mean_tiou_all = np.mean([r["iou"] for r in all_q_results if r["label"] == 1 and r["gold_span"] is not None])
    acc_all = np.mean([r["is_correct"] for r in all_q_results])
    score_all = 0.4 * acc_all + 0.6 * mean_tiou_all
    
    fold_mean_tious = {f: np.mean(t) for f, t in fold_tious.items()}
    fold_mean_accs = {f: np.mean(a) for f, a in fold_accs.items()}
    fold_scores = {f: 0.4 * fold_mean_accs[f] + 0.6 * fold_mean_tious[f] for f in folds}
    
    return {
        "overall_score": score_all,
        "overall_accuracy": acc_all,
        "overall_mean_tiou": mean_tiou_all,
        "fold_tious": fold_mean_tious,
        "fold_accs": fold_mean_accs,
        "fold_scores": fold_scores,
        "results": all_q_results
    }

def check_promotion_gate(base_eval, exp_eval):
    """
    Checks promotion gate:
    - mean delta tIoU >= +0.015
    - total score up
    - improvement in >= 4 of 5 folds
    - bootstrap over conversations (1000 resamples) delta > 0 in >= 90% of resamples
    - accuracy may not fall by more than 1 question
    """
    delta_tiou = exp_eval["overall_mean_tiou"] - base_eval["overall_mean_tiou"]
    delta_score = exp_eval["overall_score"] - base_eval["overall_score"]
    
    base_correct = sum([r["is_correct"] for r in base_eval["results"]])
    exp_correct = sum([r["is_correct"] for r in exp_eval["results"]])
    delta_correct = exp_correct - base_correct
    
    improved_folds = 0
    for f in base_eval["fold_tious"]:
        if exp_eval["fold_tious"][f] > base_eval["fold_tious"][f]:
            improved_folds += 1
            
    # Bootstrap
    conv_ids = sorted(list(set([r["conv_id"] for r in exp_eval["results"]])))
    np.random.seed(42)
    boot_deltas = []
    
    exp_by_conv = {}
    base_by_conv = {}
    for r in exp_eval["results"]:
        exp_by_conv.setdefault(r["conv_id"], []).append(r)
    for r in base_eval["results"]:
        base_by_conv.setdefault(r["conv_id"], []).append(r)
        
    for _ in range(1000):
        sample_convs = np.random.choice(conv_ids, size=len(conv_ids), replace=True)
        exp_tious = []
        base_tious = []
        for c in sample_convs:
            for r in exp_by_conv[c]:
                if r["label"] == 1 and r["gold_span"] is not None:
                    exp_tious.append(r["iou"])
            for r in base_by_conv[c]:
                if r["label"] == 1 and r["gold_span"] is not None:
                    base_tious.append(r["iou"])
        boot_deltas.append(np.mean(exp_tious) - np.mean(base_tious))
        
    boot_pass_rate = np.mean([d > 0 for d in boot_deltas])
    
    passed = (
        delta_tiou >= 0.015 and
        delta_score > 0 and
        improved_folds >= 4 and
        boot_pass_rate >= 0.90 and
        delta_correct >= -1
    )
    
    print("=== PROMOTION GATE CHECK ===")
    print(f"Delta Mean tIoU: {delta_tiou:+.4f} (Required: >= +0.015)")
    print(f"Delta Total Score: {delta_score:+.4f} (Required: > 0)")
    print(f"Improved Folds: {improved_folds}/5 (Required: >= 4)")
    print(f"Bootstrap Win Rate: {boot_pass_rate:.1%} (Required: >= 90.0%)")
    print(f"Delta Correct Questions: {delta_correct} (Required: >= -1)")
    print(f"OVERALL PROMOTION DECISION: {'PASSED' if passed else 'REJECTED'}")
    
    return passed
