import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import pandas as pd
import re
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv

import time

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from config.config import EMBEDDING_MODEL
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from evaluation.rag import RAGEval
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from evaluation.build_eval_dataset_chekpointed import build_eval_dataset

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

# Load Dataset
df = pd.read_excel(r"D:\Project AI\TanyaLPDP\Data Evaluasi TanyaLPDP.xlsx")

class Evaluation:
    def __init__(self):
        self.rag_client = RAGEval()

        ollama_llm = ChatOllama(
            model="qwen2.5:7b-instruct-q4_K_M",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=300.0,
            num_ctx=8192,
    
        )

        self.evaluator_llm = LangchainLLMWrapper(ollama_llm)

        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    def build_eval_dataset (self):
        return build_eval_dataset(self.rag_client, df, base_sleep=15.0)

    def ragas_evaluation(self, dataset):

        df_dataset = dataset.to_pandas()
        if "answer" in df_dataset.columns:
            df_dataset["answer"] = df_dataset["answer"].astype(str).apply(
                lambda x: re.sub(r'<think>.*?</think>', '', x, flags=re.DOTALL).strip()
            )

        # Reconvert ke HuggingFace Dataset
        clean_dataset = Dataset.from_pandas(df_dataset)

        run_config = RunConfig(
            max_workers=1,
            timeout=600,
            max_retries=15,
            max_wait=180
        )

        result = evaluate(
            dataset=clean_dataset,
            metrics=[
                faithfulness,
                context_recall,
                context_precision,
                answer_relevancy,
            ],
            embeddings=self.evaluator_embeddings,
            llm=self.evaluator_llm,
            run_config=run_config
        )

        return result


if __name__ == "__main__":

    eval_runner = Evaluation()

    dataset = eval_runner.build_eval_dataset()


    # # Memuat dataset 40 pertanyaan yang SUDAH BERHASIL di-generate sebelumnya
    # print("Memuat dataset dari disk...")
    # dataset = load_from_disk("evaluation/eval_dataset")

    print(
        "Menjalankan evaluasi RAGAS (menggunakan qwen2.5:7b-instruct-q4_K_M)..."
    )
    result = eval_runner.ragas_evaluation(dataset)

    # Simpan hasil evaluasi akhir ke CSV
    result_df = result.to_pandas()
    result_df.to_csv("evaluation/ragas_results_qwen2.5-7b.csv", index=False)

    print("\n=== HASIL EVALUASI RAGAS BERHASIL ===")
    print(result_df[["faithfulness", "context_recall", "context_precision", "answer_relevancy"]])

 
    # eval_runner = Evaluation()
 
    # print("Memuat dataset dari disk...")
    # dataset = load_from_disk("evaluation/eval_dataset")
 
    # sample_indices = [0, 1, 2, 3, 5, 6, 7, 17, 21]
 
    # sample_dataset = dataset.select(sample_indices)
 
    # print(f"Menjalankan evaluasi RAGAS pada {len(sample_indices)} sample dulu...")
    # result = eval_runner.ragas_evaluation(sample_dataset)
 
    # result_df = result.to_pandas()
    # result_df.to_csv("evaluation/ragas_results_SAMPLE_test.csv", index=False)
 
    # print("\n=== HASIL EVALUASI SAMPLE ===")
    # print(result_df[["faithfulness", "context_recall", "context_precision", "answer_relevancy"]])