from langchain_chroma import Chroma


def build_vector_store(documents, embedding_model, text_splitter, persist_directory):
    """
    Memotong dokumen menjadi chunks, lalu menyimpannya ke Chroma.
    """
    vector_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    chunks = text_splitter.split_documents(documents)
    print(f"[Vector Store] Jumlah chunk: {len(chunks)}")

    vector_db.add_documents(chunks)
    print("[Vector Store] Vector DB berhasil dibangun dan disimpan.")

    return vector_db
