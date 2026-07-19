from fastapi import FastAPI, HTTPException
from api.schemas import QuestionRequest, AnswerResponse
from pipeline.pipeline_rag import RAGPipeline


app = FastAPI(
    title="Tanya LPDP",
    description="REST API untuk chatbot RAG berbasis dokumen resmi LPDP.",
    version="1.1.0"
)
pipeline = RAGPipeline()

@app.get("/")
def root():
    return {"message": "Welcome to TanyaLPDP API"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Menerima pertanyaan dari pengguna dan mengembalikan jawaban
    berdasarkan dokumen resmi LPDP beserta referensinya.
    """

    try:
        results = pipeline.ask(request.question)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server: {str(e)}")

    