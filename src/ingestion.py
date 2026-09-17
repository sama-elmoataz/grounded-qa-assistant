from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    PDF_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ============================================================
# Load PDF
# ============================================================

def load_documents(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# ============================================================
# Split Documents
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# Create Embeddings
# ============================================================

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# ============================================================
# Build Vector Store
# ============================================================

def build_vectorstore(
    chunks,
    embeddings,
):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vectorstore


# ============================================================
# Save Vector Store
# ============================================================

def save_vectorstore(vectorstore):

    vectorstore.save_local(
        VECTORSTORE_PATH
    )


# ============================================================
# Main Ingestion Pipeline
# ============================================================

def main():

    print("Loading NovaDesk PDF...")

    documents = load_documents(
        PDF_PATH
    )

    print(
        f"Loaded {len(documents)} pages."
    )


    print("Splitting documents...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    print("Loading embedding model...")

    embeddings = create_embeddings()


    test_vector = embeddings.embed_query(
        "What are the NovaDesk subscription plans?"
    )

    print(
        f"Embedding dimension: "
        f"{len(test_vector)}"
    )


    print(
        "Building FAISS vector store..."
    )

    vectorstore = build_vectorstore(
        chunks,
        embeddings,
    )


    print("Saving vector store...")

    save_vectorstore(
        vectorstore
    )


    print(
        "NovaDesk ingestion completed successfully."
    )


if __name__ == "__main__":

    main()