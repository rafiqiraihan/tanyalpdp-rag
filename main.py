import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from pipeline.pipeline_rag import RAGPipeline

def main():
    print("=" * 70)
    print("[TanyaLPDP] Memulai Okestrasi Sistem Chatbot RAG...")
    print("=" * 70)

    try:
        pipeline = RAGPipeline()
        print("[STATUS] Chatbot TanyaLPDP Aktif!")
        print("Silahkan ketik pertanyaan seputar Beasiswa LPDP.")
        print("(Ketik 'keluar' atau 'exit' untuk menyudahi percakapan\n)")
        print("-" * 70)

        # Loop Obrolan Iteraktif Terminal
        while True:
            try:
                user_query = input("\nUser: ")

                # # Cek command keluar
                if user_query.strip().lower() in ['keluar', 'exit']:
                    print("\n TanyaLPDP: Terima Kasih! Semoga sukses dengan pendaftaran LPDP Anda. Sampai Jumpa!")
                    break
                
                # Lewati jika user tidak mengetik apa-apa
                if not user_query.strip():
                    continue

                print("\nTanyaLPDP sedang berpikir dan membaca dokumen...")

                # Jalankan Pipeline RAG
                result = pipeline.ask(user_query)
                
                print(f"\nChatbot TanyaLPDP:\n{result['answer']}")

                print("\nReferensi:")
                for source in result['sources']:
                    print(f"- {source['source']} (Hal. {source['page']})")
                
                print("\n" + "-" * 70)


            except KeyboardInterrupt:
                print("\n\nProgram diberhentikan paksa. Sampai Jumpa!")
                break

            except Exception as e:
                print(f"\nTerjadi Kesalahan: {e}")


    except Exception as e:
        print(f"\n [CRASH] Gagal menyalakan chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


    