import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from ingestion.data_loader import load_data
from indexing.chunking import get_text_splitter
from indexing.embedding import get_embedding_model
from indexing.vector_store import build_vector_store

import shutil

def reset_vector_db(path):
    shutil.rmtree(path, ignore_errors=True)

def run_indexing_pipeline():

    path_raw = os.path.join(ROOT_DIR, "data", "raw")
    path_db = os.path.join(ROOT_DIR, "data", "vector_db")

    # --- TAHAP 1: LOAD PDF ---
    try:
        documents = load_data(path_raw)

        print("=" * 80)
        print(f"Jumlah Document : {len(documents)}")
            
        if not documents:
            print("[Error Loader]: File PDF tidak ditemukan.")
            return
        
        # Validasi Intip Teks agar terhindar dari Silent Error
        print(f" [OK] Berhasil Memuat {len(documents)} halaman.")
    except Exception as e:
        print(f"[CRASH Loader]: {e}")
        return
    
    # --- TAHAP 2: INITIATE SPLITTER ---
    try:
        text_splitter = get_text_splitter()
        print(f"[OK] Splitter chunk siap.")
    except Exception as e:
        print(f"[CRASH Chunking]: {e}")
        return
    
    # --- TAHAP 3: LOAD EMBEDDING
    try:
        embedding_model = get_embedding_model()
        print(f"[OK] Model embedding siap di memori.")
    except Exception as e:
        print(f"[CRASH Embeddding]: {e}")
        return

    # --- TAHAP 4: BUILD & SAVE VECTOR STORE ---
    try:
        vector_db = build_vector_store(
            documents=documents,
            embedding_model=embedding_model,
            text_splitter=text_splitter,
            persist_directory=path_db
        )

        # --- UJI COBA RETRIEVAL LANGSUNG ---
        print("\n [Self-Test] Menguji pencarian database...")
        test_query = "Syarat beasiswa LPDP"
        retriever_docs = vector_db.similarity_search(test_query, k=10)

        if retriever_docs:
            print(f"Berhasil menarik {len(retriever_docs)} chunk konteks.")
            print("\n [Pipeline] 100% SUKSES! Database siap digunakan untuk Chatbot.")
        else:
            print("[Warning]: Database terbentuk tapi tidak mengembalikan hasil pencarian.")
    
    except Exception as e:
        print(f"[CRASH Vector Store]: {e}")
        return
    
    
if __name__ == "__main__":
    reset_vector_db("data/vector_db")
    run_indexing_pipeline()

