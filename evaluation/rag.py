from retrieval.searcher import AdvancedLPDPSearcher
from generation.generator import LPDPGenerator


class RAGEval:
    def __init__(self):
        self.searcher = AdvancedLPDPSearcher()
        self.retriever = self.searcher.final_pipeline
        
        self.generator = LPDPGenerator()
        
        self.rag_chain = self.generator.create_rag_chain()

    def run_rag(self, query: str) -> dict:
        """
        Menerima pertanyaan, memprosesnya melalui RAG, dan mengembalikan jawaban dan context.
        """

        docs = self.retriever.invoke(query)
        retrieval_contexts = [doc.page_content for doc in docs]
        context = self.generator._format_docs(docs)
        
        response = self.rag_chain.invoke({
            "context": context,
            "question": query
        })

        
        return {
            "answer": response,
            "retrieval_contexts": retrieval_contexts
        }


if __name__ == "__main__":
    rag = RAGEval()
    query_test = "Seperti apa skema Beasiswa Akselerasi Magister?"
    results = rag.run_rag(query_test)

    print(results)