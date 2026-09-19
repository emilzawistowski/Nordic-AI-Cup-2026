"""Comprehensive benchmark runner for Medical Appointment models and pipelines.

Warming up models prior to timing and enforcing strict monotonic deadline checking.
Records complete metrics to experiments/runs.jsonl.
"""

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
import numpy as np

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import (
    gold_evidence,
    group_questions_by_conversation,
    load_sample_audio,
    temporal_iou,
    evidence_interval,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")


def run_benchmark(predict_fn, run_id="exp_baseline", hypothesis="Baseline evaluation", warm_up=True):
    # Load splits
    splits_file = Path("experiments/splits.json")
    if splits_file.exists():
        with open(splits_file) as f:
            splits = json.load(f)
    else:
        splits = None

    conversations = group_questions_by_conversation()
    
    # Warmup
    if warm_up and conversations:
        logger.info("Warming up pipeline with first conversation...")
        first_conv_id, first_rows = conversations[0]
        audio_bytes = load_sample_audio(first_conv_id)
        req = ASRQuestionRequestDto(
            audio_base64=base64.b64encode(audio_bytes).decode('ascii'),
            audio_filename=first_conv_id,
            questions=[r['question'] for r in first_rows],
        )
        try:
            _ = predict_fn(req)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    # Benchmark loop
    all_latencies = []
    total_questions = 0
    correct_questions = 0
    unanswered_errors = 0
    by_type = {"positive": [0, 0], "hard_negative": [0, 0], "off_topic": [0, 0]}
    
    gold_positive_tious = []
    conditional_tious = []
    missing_spans = 0
    timeouts = 0
    malformed_responses = 0

    attempt_start_mono = time.monotonic()
    attempt_budget_s = len(conversations) * 60.0
    consecutive_timeouts = 0

    logger.info(f"Starting benchmark over {len(conversations)} conversations...")

    for conv_idx, (audio_filename, rows) in enumerate(conversations):
        # Monotonic deadline check per conversation budget
        if time.monotonic() - attempt_start_mono > attempt_budget_s:
            logger.error("Attempt budget exceeded! Treating remaining tail questions as unanswered/error.")
            for r in rows:
                total_questions += 1
                unanswered_errors += 1
                qtype = r['question_type']
                by_type[qtype][1] += 1
                if int(r['label']) == 1:
                    gold_positive_tious.append(0.0)
                    missing_spans += 1
            continue

        if consecutive_timeouts >= 5:
            logger.error("5 consecutive timeouts! Failing remaining tail questions.")
            for r in rows:
                total_questions += 1
                unanswered_errors += 1
                qtype = r['question_type']
                by_type[qtype][1] += 1
                if int(r['label']) == 1:
                    gold_positive_tious.append(0.0)
                    missing_spans += 1
            continue

        questions = [r['question'] for r in rows]
        audio_bytes = load_sample_audio(audio_filename)
        req = ASRQuestionRequestDto(
            audio_base64=base64.b64encode(audio_bytes).decode('ascii'),
            audio_filename=audio_filename,
            questions=questions,
        )

        conv_start_mono = time.monotonic()
        timed_out = False
        try:
            # 60 second monotonic deadline per request
            res = predict_fn(req)
            elapsed_s = time.monotonic() - conv_start_mono
            if elapsed_s > 60.0:
                timed_out = True
                timeouts += 1
                consecutive_timeouts += 1
                logger.error(f"Conversation {audio_filename} exceeded 60s deadline ({elapsed_s:.2f}s).")
            else:
                consecutive_timeouts = 0
        except Exception as exc:
            elapsed_s = time.monotonic() - conv_start_mono
            logger.exception(f"Request exception on {audio_filename}: {exc}")
            malformed_responses += 1
            res = None

        all_latencies.append(elapsed_s)

        if res is None or timed_out or not isinstance(res, ASRQuestionResponseDto) or len(res.answers) != len(questions):
            malformed_responses += 1 if res is not None and not timed_out else 0
            # Fail all questions in this conversation
            for r in rows:
                total_questions += 1
                unanswered_errors += 1
                qtype = r['question_type']
                by_type[qtype][1] += 1
                if int(r['label']) == 1:
                    gold_positive_tious.append(0.0)
                    missing_spans += 1
            continue

        # Score questions
        for idx, r in enumerate(rows):
            total_questions += 1
            label = int(r['label'])
            qtype = r['question_type']
            by_type[qtype][1] += 1

            pred_ans = bool(res.answers[idx])
            start_t = res.evidence_start[idx]
            end_t = res.evidence_end[idx]
            pred_span = evidence_interval(start_t, end_t)
            gold_span = gold_evidence(r)

            if pred_ans == bool(label):
                correct_questions += 1
                by_type[qtype][0] += 1

            if label == 1 and gold_span is not None:
                iou = temporal_iou(gold_span, pred_span)
                gold_positive_tious.append(iou)
                if pred_span is None:
                    missing_spans += 1
                if pred_ans:
                    conditional_tious.append(iou)

    # Compute metrics
    accuracy = correct_questions / total_questions if total_questions else 0.0
    mean_tiou = float(np.mean(gold_positive_tious)) if gold_positive_tious else 0.0
    cond_tiou = float(np.mean(conditional_tious)) if conditional_tious else 0.0
    combined_score = 0.4 * accuracy + 0.6 * mean_tiou

    pos_rec = by_type["positive"][0] / by_type["positive"][1] if by_type["positive"][1] else 0.0
    lat_p50 = float(np.percentile(all_latencies, 50)) if all_latencies else 0.0
    lat_p95 = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    lat_max = float(max(all_latencies)) if all_latencies else 0.0

    per_type_acc = {
        k: (v[0] / v[1] if v[1] else 0.0) for k, v in by_type.items()
    }

    metrics = {
        "run_id": run_id,
        "git_sha": os.popen("git log -n 1 --format='%H'").read().strip(),
        "hypothesis": hypothesis,
        "seed": 42,
        "hardware": "Apple Silicon (MLX)",
        "accuracy": accuracy,
        "mean_tiou": mean_tiou,
        "conditional_tiou": cond_tiou,
        "combined_score": combined_score,
        "positive_recall": pos_rec,
        "per_type_accuracy": per_type_acc,
        "missing_spans": missing_spans,
        "p50_latency_s": lat_p50,
        "p95_latency_s": lat_p95,
        "max_latency_s": lat_max,
        "timeouts": timeouts,
        "malformed_responses": malformed_responses,
        "timestamp": time.time(),
    }

    # Append to runs.jsonl
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/runs.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")

    logger.info(f"=== BENCHMARK RESULT [{run_id}] ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Mean tIoU: {mean_tiou:.4f}")
    logger.info(f"Combined Score: {combined_score:.4f}")
    logger.info(f"Per-type Acc: {per_type_acc}")
    logger.info(f"Latency P50: {lat_p50:.2f}s, P95: {lat_p95:.2f}s, Max: {lat_max:.2f}s")
    logger.info(f"Timeouts: {timeouts}, Malformed: {malformed_responses}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp1_baseline")
    parser.add_argument("--hypothesis", default="Baseline evaluation with full instrumentation")
    args = parser.parse_args()

    from example import predict
    run_benchmark(predict, run_id=args.run_id, hypothesis=args.hypothesis)
