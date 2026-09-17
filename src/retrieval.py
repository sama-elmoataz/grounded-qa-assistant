from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    RETRIEVAL_K,
)


def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


def load_vectorstore(embeddings):

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore



def retrieve_documents(
    query,
    vectorstore,
):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )

    documents = retriever.invoke(
        query
    )

    return documents