# Medical Appointment Optimization Final Summary

- **Final Local Evaluator Score**: **0.6394** (Accuracy: 0.9744, Mean tIoU: 0.4161)

## Best Approach Summary
The optimal solution transcribes conversation audio once using MLX Whisper (`whisper-large-v3-turbo`) with word-level timestamps, then evaluates all 10 questions concurrently in a single structured prompt call using `Qwen3-4B-Instruct-2507-4bit`. This batching approach reduced total QA processing time by 53.5% (down to ~10.2s P50 latency) while improving candidate evidence selection through token-aligned window matching.

## Working Solution Files
- [example.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/example.py)
- [medical_asr.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/medical_asr.py)
- [medical_reasoner.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/medical_reasoner.py)
- [medical_evidence.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/medical_evidence.py)
- [fact_verifier.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/fact_verifier.py)
- [generate_splits.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/generate_splits.py)
- [benchmark.py](file:///Users/emil/Desktop/Nordic%20AI%20Cup/Nordic-AI-Cup-2026/medical-appointment/benchmark.py)

## Required Model Weights
- `mlx-community/whisper-large-v3-turbo` (Cached under `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo`)
- `mlx-community/Qwen3-4B-Instruct-2507-4bit` (Cached under `~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit`)

## Dependencies
- Standard `requirements.txt` dependencies: `fastapi`, `uvicorn`, `pydantic`, `requests`, `mlx-whisper`, `mlx-lm`, `numpy`. No new dependencies were added to `requirements.txt`.

## Dead Ends / What Did NOT Work
1. **Local Rule-Based Fact Verifier (`exp4_fact_verifier`)**: Pre-filtering hard-negative mismatches using deterministic entity/dose regex rules did not yield incremental accuracy gains over Qwen3-4B batching, and fell short of the +0.01 promotion gate.
2. **Guarded Negation Alignment (`exp2_evidence_alignment`)**: Strict filtering of evidence candidate windows based on negation token presence improved tIoU slightly (+0.0024), but failed the strict promotion gate threshold (+0.015).
