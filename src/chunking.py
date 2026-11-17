import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_and_split_documents(data_dir: str = "data/documents", chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Loads all supported documents from the data directory and splits them 
    into semantic chunks with detailed enterprise metadata.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Directory {data_dir} created. Please place your enterprise documents there.")
        return []

    documents = []
    
    # Iterate through documents in the folder
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Load PDF documents
        if filename.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["file_type"] = "PDF"
                documents.extend(docs)
                print(f"Successfully loaded PDF: {filename} ({len(docs)} pages)")
            except Exception as e:
                print(f"Error loading PDF {filename}: {e}")
                
        # Load Text / Markdown documents
        elif filename.lower().endswith((".txt", ".md")):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["file_type"] = "Text"
                documents.extend(docs)
                print(f"Successfully loaded text document: {filename}")
            except Exception as e:
                print(f"Error loading text file {filename}: {e}")

    if not documents:
        print("No documents found to process.")
        return []

    # Configure advanced recursive text splitting for semantic boundary preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    chunked_documents = text_splitter.split_documents(documents)
    
    # Assign unique chunk IDs for enterprise auditability
    for idx, chunk in enumerate(chunked_documents):
        chunk.metadata["chunk_id"] = f"chunk_{idx}"
        # Ensure page metadata exists
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0

    print(f"Total raw document pages/sections: {len(documents)}")
    print(f"Generated {len(chunked_documents)} optimized semantic chunks (Chunk Size: {chunk_size}, Overlap: {chunk_overlap}).")
    
    return chunked_documents