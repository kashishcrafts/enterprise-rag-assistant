import os
import time
import google.generativeai as genai

# Configure API key from environment
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def generate_rag_answer(query: str, retrieved_docs_with_scores: list, model_name: str = "gemini-1.5-flash", max_retries: int = 3) -> dict:
    """
    Generates a grounded RAG response using Google Gemini, incorporating retrieved context
    and source metadata citations with built-in exponential backoff rate-limit handling.
    """
    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY environment variable is missing. Please set your API key.",
            "sources": []
        }

    if not retrieved_docs_with_scores:
        return {
            "answer": "I could not find any relevant information in the uploaded documents to answer your query.",
            "sources": []
        }

    # Format context and extract unique sources
    context_blocks = []
    sources = set()
    
    for item in retrieved_docs_with_scores:
        # Handle both (doc, score) tuples and raw doc objects
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc, score = item, 0.0
            
        source_name = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", 0) + 1
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        
        sources.add(f"{source_name} (Page {page_num})")
        context_blocks.append(f"[Source: {source_name} | Page {page_num} | ID: {chunk_id}]\n{doc.page_content}")

    combined_context = "\n\n---\n\n".join(context_blocks)

    # Enterprise prompt engineering with strict anti-hallucination guardrails
    prompt = f"""You are an expert Enterprise Document Intelligence & RAG Assistant. 
Answer the user's query professionally and accurately based ONLY on the provided context below. 
If the answer cannot be found in the context, state clearly that the information is not available in the documents. 
Do not make up facts or extrapolate beyond the provided text. Always maintain a professional tone.

Context Documents:
{combined_context}

User Query: {query}

Professional Answer:"""

    # Call Gemini with retry and exponential backoff for rate limits (HTTP 429)
    model = genai.GenerativeModel(model_name)
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            answer_text = response.text if response and response.text else "No response generated."
            return {
                "answer": answer_text,
                "sources": list(sources)
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** (attempt + 1)
                    print(f"Rate limit reached (429). Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
            return {
                "answer": f"Error generating answer from LLM after retries: {error_str}",
                "sources": list(sources)
            }
            
    return {
        "answer": "Error: Maximum retry limit reached due to API rate limits.",
        "sources": list(sources)
    }