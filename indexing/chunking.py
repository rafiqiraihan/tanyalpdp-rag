from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter():
    """
    Menyiapkan splitter untuk chunk yang akan disimpan langsung ke Chroma.
    """
    print(f"[Chunking] Menyiapkan Splitter -> Chunk: {CHUNK_SIZE}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    return text_splitter