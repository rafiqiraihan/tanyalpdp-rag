from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

def load_data (folder_path):
    print(f"Memulai pemindaian folder: {folder_path}")
    
    # Inisialisasi untuk membaca folder
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    # Eksekusi pembacaan
    documents = loader.load()

    print(f"Berhasil memuat {len(documents)} halaman dokumen.")

    return documents


