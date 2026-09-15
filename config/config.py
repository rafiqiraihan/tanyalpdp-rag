import torch

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL = "BAAI/bge-m3"

RERANKER_MODEL = "BAAI/bge-reranker-base"

VECTOR_SEARCH_K = 10
BM25_K = 10
RERANK_TOP_N = 5

LLM_MODEL = "openai/gpt-oss-120b"

TEMPERATURE = 0