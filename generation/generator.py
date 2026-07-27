import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from llm.prompt_builder import get_lpdp_prompt_template
from config.config import LLM_MODEL, TEMPERATURE

load_dotenv()

class LPDPGenerator:
    
    def __init__(self):

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY tidak ditemukan di file .env")
        
        # Inisialisasi Model Llama3 70B via Groq
        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.prompt_template = get_lpdp_prompt_template()
        self.output_parser = StrOutputParser()
        self.source = os.path.basename

    def _format_docs(self, docs):
        """Menggabungkan teks dari dokumen-dokumen yang ditemukan menjadi satu konteks utuh"""
       
        formatted_docs = []

        for i, doc in enumerate(docs, start=1):
            formatted_docs.append(
                f"""======== Dokumen {i} ========

    Isi:
    {doc.page_content}
    """
            )

        return "\n\n".join(formatted_docs)
    
    def _extract_sources(self, docs):
        """
        Mengambil metadata sumber dari dokumen hasil retrieval.
        """
        seen = set()
        sources = []

        for doc in docs:
            source = self.source(doc.metadata["source"])
            page = doc.metadata["page"]

            reference = (source, page)
            if reference not in seen:

                seen.add(reference)

                sources.append({
                    "source" : source,
                    "page": page
                })

        return sources

    
    def create_rag_chain(self):
        """
        Merakit Pipeline RAG Jalur LangChain Expression Language (LCEL).
        Menerima input 'retriever' yang berasal dari searcher.py
        """

        rag_chain = (
            self.prompt_template
            | self.llm
            | self.output_parser
        )

        return rag_chain