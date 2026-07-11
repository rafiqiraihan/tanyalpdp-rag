from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL

def get_embedding_model():
    """
    Menginisialisasi dan mengambil model embedding untuk mengubah teks menjadi vector.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'}
        )

        return embeddings
    
    except Exception as e:
        raise e