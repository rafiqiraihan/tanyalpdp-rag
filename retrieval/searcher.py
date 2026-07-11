# retrieval/searcher.py
import os
import sys

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from indexing.embedding import get_embedding_model
from config.config import VECTOR_SEARCH_K, BM25_K, RERANKER_MODEL, RERANK_TOP_N


class AdvancedLPDPSearcher:
    def __init__(self):
       
        self.path_vector_db = os.path.join(ROOT_DIR, "data", "vector_db")

        self.embedding_model = get_embedding_model()
        self.vector_db = Chroma(
            persist_directory=self.path_vector_db,
            embedding_function=self.embedding_model,
        )

        self.final_pipeline = None
        self._setup_hybrid_and_reranker()

    def _load_chroma_documents(self) -> list[Document]:
        stored_data = self.vector_db.get(include=["documents", "metadatas"])
        texts = stored_data.get("documents", [])
        metadatas = stored_data.get("metadatas", [])

        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas)
            if text
        ]

    def _setup_hybrid_and_reranker(self):
       
        chroma_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_K}
        )
        base_retriever = chroma_retriever

        chunk_documents = self._load_chroma_documents()
        if chunk_documents:
            bm25_retriever = BM25Retriever.from_documents(chunk_documents)
            bm25_retriever.k = BM25_K

            base_retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, bm25_retriever],
                weights=[0.7, 0.3],
            )
        else:
            raise RuntimeError()

        rerank_model = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            model_kwargs={"device": "cuda"},
        )
        reranker = CrossEncoderReranker(model=rerank_model, top_n = RERANK_TOP_N)

        self.final_pipeline = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )

    def search(self, query: str) -> list[Document]:
        return self.final_pipeline.invoke(query)


if __name__ == "__main__":
    searcher = AdvancedLPDPSearcher()

    query_test = "Berapa batas usia maksimal mendaftar beasiswa LPDP Magister?"
    results = searcher.search(query_test)

    print("\n============= HASIL RERANKING AKHIR =============")
    for i, doc in enumerate(results):
        print(f"\n[Chunk Ke-{i + 1}] (Ukuran Karakter: {len(doc.page_content)})")
        print("Metadata:", doc.metadata)
        print(f"Isi Chunk:\n{doc.page_content[:400]}...")
        print("-" * 60)
