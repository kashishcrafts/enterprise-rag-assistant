import os
import sys

# Add parent directory to path so absolute imports work properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever import load_vector_store, retrieve_relevant_chunks
from embeddings import get_embedding_model
from rag_pipeline import generate_rag_answer

# Define a small evaluation dataset (Questions, Expected Keywords in Answer, Expected Source)
EVAL_DATASET = [
    {
        "query": "What is the policy for working from home?",
        "expected_keywords": ["remote work", "remote work days", "manager"],
        "expected_source": "HR_Policy_Handbook.pdf"
    },
    {
        "query": "What is the notice period for resignation?",
        "expected_keywords": ["notice period", "probationary"],
        "expected_source": "HR_Policy_Handbook.pdf"
    },
    {
        "query": "How many annual leave days do employees get?",
        "expected_keywords": ["leave", "days"],
        "expected_source": "HR_Policy_Handbook.pdf"
    }
]

def run_evaluation():
    """
    Runs the RAG evaluation dataset against the local vector store and LLM pipeline,
    printing metrics for retrieval match and answer correctness.
    """
    vector_db_path = "vectorstore/faiss_index"
    if not os.path.exists(vector_db_path):
        print("Error: Vector database index not found. Please index your documents first via the Streamlit app.")
        return

    print("Initializing embedding model and loading vector store for evaluation...")
    emb_model = get_embedding_model()
    vectorstore = load_vector_store(emb_model, vector_db_path)

    total_tests = len(EVAL_DATASET)
    retrieval_passed = 0
    answer_passed = 0

    print(f"\nRunning RAG Evaluation on {total_tests} test cases...\n" + "="*60)

    for idx, test in enumerate(EVAL_DATASET, 1):
        query = test["query"]
        print(f"\n[Test {idx}/{total_tests}] Query: '{query}'")

        # 1. Retrieve chunks
        retrieved = retrieve_relevant_chunks(vectorstore, query, top_k=3)
        
        # Check retrieval source match
        source_matched = False
        for doc, score in retrieved:
            if test["expected_source"] in doc.metadata.get("source", ""):
                source_matched = True
                break
        
        if source_matched:
            retrieval_passed += 1
            print("  -> Retrieval Check: PASSED (Found expected source)")
        else:
            print("  -> Retrieval Check: FAILED (Expected source not in top-k chunks)")

        # 2. Generate RAG Answer
        result = generate_rag_answer(query, retrieved)
        answer_text = result["answer"].lower()

        # Check answer correctness (keyword match)
        keywords_found = all(kw.lower() in answer_text for kw in test["expected_keywords"])
        if keywords_found:
            answer_passed += 1
            print("  -> Answer Quality Check: PASSED (Expected keywords found)")
        else:
            print("  -> Answer Quality Check: FAILED (Keywords missing)")
            print(f"     Generated Answer: {result['answer'][:150]}...")

    print("\n" + "="*60)
    print(f"EVALUATION COMPLETE:")
    print(f"- Retrieval Accuracy: {retrieval_passed}/{total_tests} ({retrieval_passed/total_tests*100:.1f}%)")
    print(f"- Answer Correctness: {answer_passed}/{total_tests} ({answer_passed/total_tests*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()