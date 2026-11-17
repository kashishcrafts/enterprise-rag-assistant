import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Scans the specified directory for PDF and DOCX files and loads them
    into LangChain Document objects with source metadata.
    """
    documents = []
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return documents

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        elif file.endswith(".docx"):
            print(f"Loading Word Document: {file}")
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
            
    print(f"Successfully loaded {len(documents)} pages/sections from documents.")
    return documents

if __name__ == "__main__":
    docs = load_documents_from_directory("data/documents")
    print(f"Sample text from first doc: {docs[0].page_content[:200] if docs else 'No documents found.'}")