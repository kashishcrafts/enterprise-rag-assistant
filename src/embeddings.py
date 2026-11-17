from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Initializes and returns a HuggingFace sentence transformer embedding model.
    """
    print(f"Loading embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

if __name__ == "__main__":
    model = get_embedding_model()
    test_vector = model.embed_query("What is the company leave policy?")
    print(f"Embedding generated successfully. Vector dimension: {len(test_vector)}")