import json
import os
import time
import re
import pandas as pd
from datasets import Dataset

CHECKPOINT_PATH = "evaluation/checkpoint.jsonl"

def load_checkpoint() -> dict:
    """
    Muat hasil yang sudah pernah diproses, key = pertanyaan.
    """

    done = {}

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8",) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[row["question"]] = row
    return done

def append_checkpoint(row: dict):
    """
    Simpan satu hasil baru langsung ke disk (append, bukan overwrite)
    """

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def clean_answer(raw_answer: str) -> str:
    if "<think>" in raw_answer  and "</think>" in raw_answer:
        return re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    elif "<think>" in raw_answer:
        return ""
    return raw_answer.strip()

def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "rate_limit" in msg
        or "rate limit" in msg
        or "429" in msg
        or "quota" in msg
        or "tokens per day" in msg
        or "tokens per minute" in msg
    )


def build_eval_dataset(rag_client, df, base_sleep: float = 15.0):
    questions = df["Question"].tolist()
    ground_truths = df["Expected Answer"].tolist()
 
    done = load_checkpoint()
    print(f"Ditemukan {len(done)} pertanyaan yang sudah selesai sebelumnya, akan di-skip.")
 
    for idx, (q, gt) in enumerate(zip(questions, ground_truths)):
        if q in done:
            print(f"[{idx + 1}/{len(questions)}] Skip (sudah ada di checkpoint): {q[:50]}...")
            continue
 
        print(f"[{idx + 1}/{len(questions)}] Memproses: {q[:50]}...")
 
        attempt = 0
        while True:
            attempt += 1
            try:
                rag = rag_client.run_rag(q)
 
                cleaned_answer = clean_answer(rag["answer"])
 
                contexts = rag["retrieval_contexts"]
                if isinstance(contexts, str):
                    contexts = [contexts]
 
                append_checkpoint({
                    "question": q,
                    "answer": cleaned_answer,
                    "contexts": contexts,
                    "ground_truth": gt,
                })
 
                time.sleep(base_sleep)
                break
 
            except Exception as e:
                if is_rate_limit_error(e):
                    wait = min(60 * attempt, 300)  # backoff, maks 5 menit
                    print(f"  Kena rate limit, tunggu {wait} detik lalu coba lagi... ({e})")
                    time.sleep(wait)
                    continue
                else:
                    print(f"  Error tidak terduga (bukan rate limit): {e}")
                    raise e

 
    all_rows = load_checkpoint()
    ordered_rows = [all_rows[q] for q in questions if q in all_rows]
 
    data_eval = Dataset.from_dict({
        "question": [r["question"] for r in ordered_rows],
        "answer": [r["answer"] for r in ordered_rows],
        "contexts": [r["contexts"] for r in ordered_rows],
        "ground_truth": [r["ground_truth"] for r in ordered_rows],
    })
    data_eval.save_to_disk("evaluation/eval_dataset")
    return data_eval

