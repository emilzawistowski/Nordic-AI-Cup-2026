import os
import json
import hashlib
from pathlib import Path
from utils import group_questions_by_conversation, load_sample_audio
from medical_asr import transcribe_audio, extract_words
from medical_reasoner import answer_questions_batch

TRANSCRIPTS_DIR = Path("transcripts")
LLM_CACHE_DIR = Path("experiments/llm_cache")

def build_cache():
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    LLM_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    
    conversations = group_questions_by_conversation()
    print(f"Building cache for {len(conversations)} conversations...")
    
    cached_transcripts = {}
    cached_llm = {}

    for audio_filename, rows in conversations:
        conv_id = audio_filename.replace("conversation_", "").replace(".mp3", "")
        t_file = TRANSCRIPTS_DIR / f"{conv_id}.json"
        
        if t_file.exists():
            with open(t_file, "r") as f:
                transcription = json.load(f)
        else:
            print(f"Transcribing {audio_filename}...")
            audio_bytes = load_sample_audio(audio_filename)
            transcription = transcribe_audio(audio_bytes, audio_filename)
            with open(t_file, "w") as f:
                json.dump(transcription, f, indent=2)
                
        transcript_text = transcription.get("text", "").strip()
        words = extract_words(transcription)
        cached_transcripts[audio_filename] = {
            "transcription": transcription,
            "transcript_text": transcript_text,
            "words": words,
            "rows": rows
        }
        
        # Cache LLM outputs for baseline prompt
        questions = [r["question"] for r in rows]
        q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        prompt_key = hashlib.md5(f"{transcript_text}\n{q_str}".encode()).hexdigest()
        llm_file = LLM_CACHE_DIR / f"{prompt_key}.json"
        
        if llm_file.exists():
            with open(llm_file, "r") as f:
                qa_results = json.load(f)
        else:
            print(f"Running LLM for {audio_filename}...")
            qa_results = answer_questions_batch(transcript_text, questions)
            with open(llm_file, "w") as f:
                json.dump(qa_results, f, indent=2)
                
        cached_llm[audio_filename] = qa_results
        
    print("Cache build complete.")
    return cached_transcripts, cached_llm

if __name__ == "__main__":
    build_cache()
