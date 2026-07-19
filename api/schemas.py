from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class SourceReference(BaseModel):
    source: str
    page: int

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceReference]