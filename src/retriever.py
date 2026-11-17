import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # Fallback or community-mapped

def create_and_save_vector_store(chunked_documents, embedding_model, save_path: str = "vectorstore/faiss_index"):
    """
    Creates a FAISS vector store from chunked documents and saves it locally.
    """
    if not chunked_documents:
        print("No documents provided to index.")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("Building FAISS vector store index...")
    vectorstore = FAISS.from_documents(chunked_documents, embedding_model)
    vectorstore.save_local(save_path)
    print(f"Vector store successfully saved to '{save_path}'.")
    return vectorstore

def load_vector_store(embedding_model, save_path: str = "vectorstore/faiss_index"):
    """
    Loads an existing FAISS vector store from local disk safely.
    """
    if not os.path.exists(save_path):
        print(f"Vector store not found at {save_path}.")
        return None
        
    print(f"Loading FAISS index from {save_path}...")
    vectorstore = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

def create_hybrid_retriever(vectorstore, chunked_documents, top_k: int = 4):
    """
    Combines FAISS semantic vector search with BM25 keyword search 
    using LangChain's EnsembleRetriever for enterprise-grade precision.
    """
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    if chunked_documents and len(chunked_documents) > 0:
        try:
            bm25_retriever = BM25Retriever.from_documents(chunked_documents)
            bm25_retriever.k = top_k
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )
            print("Hybrid search retriever (FAISS + BM25) initialized successfully.")
            return ensemble_retriever
        except Exception as e:
            print(f"Ensemble initialization notice: {e}. Falling back to vector retriever.")
            return vector_retriever
    else:
        return vector_retriever

def retrieve_relevant_chunks(vectorstore, query: str, top_k: int = 4):
    """
    Performs standard vector similarity retrieval returning documents with scores.
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    return docs_with_scores