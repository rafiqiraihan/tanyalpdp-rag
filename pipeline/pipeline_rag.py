# pipeline/rag_pipeline.py

from retrieval.searcher import AdvancedLPDPSearcher
from llm.generator import LPDPGenerator

class RAGPipeline:
    def __init__(self):
        print("[Inisialisasi Pipeline] Menyiapkan Retriever dan LLM...")
        # 1. Inisialisasi Retriever
        self.searcher = AdvancedLPDPSearcher()
        self.retriever = self.searcher.final_pipeline
        
        # 2. Inisialisasi LLM
        self.generator = LPDPGenerator()
        
        # 3. Menggabungkan Retriever dan LLM
        self.rag_chain = self.generator.create_rag_chain()
        print("[Inisialisasi Selesai] RAG Pipeline siap digunakan.")

    def ask(self, question: str) -> dict:
        """
        Menerima pertanyaan, memprosesnya melalui RAG, dan mengembalikan jawaban beserta referensi.
        """

        docs = self.retriever.invoke(question)
        context = self.generator._format_docs(docs)
        sources = self.generator._extract_sources(docs)
        
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        
        return {
            "answer": response,
            "sources": sources
        }