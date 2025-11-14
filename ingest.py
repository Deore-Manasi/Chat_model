import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"  # folder containing all your PDFs
DB_PATH = "vector_store"

def ingest_all_pdfs():
    docs = []

    # loop through all files in /data
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"📄 Loading: {file_path}")

            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)

    # text splitter
    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # embeddings (FREE — HuggingFace)
    print("🧠 Building embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # build FAISS vector DB
    print("💾 Saving FAISS vector database...")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(DB_PATH)

    print("✅ Ingestion completed! Vector DB saved at:", DB_PATH)


if __name__ == "__main__":
    ingest_all_pdfs()
