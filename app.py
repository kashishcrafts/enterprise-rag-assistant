import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modular backend functions
from src.chunking import load_and_split_documents
from src.embeddings import get_embedding_model
from src.retriever import create_and_save_vector_store, load_vector_store, retrieve_relevant_chunks
from src.rag_pipeline import generate_rag_answer

# Page Configuration
st.set_page_config(
    page_title="Enterprise Document Intelligence & RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Enterprise Styling (CSS)
st.markdown("""
<style>
    /* Global App Background & Font */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Headers */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    /* Source Citation Cards */
    .source-card {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric / Status Boxes */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Control Panel & Document Management ---
with st.sidebar:
    st.markdown("### ⚡ **Enterprise AI Control**")
    st.markdown("---")
    
    st.markdown("#### 📁 Document Repository")
    data_dir = "data/documents"
    os.makedirs(data_dir, exist_ok=True)
    
    uploaded_file = st.file_uploader("Upload PDF or Text Document", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        file_path = os.path.join(data_dir, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")
            st.rerun()

    files = os.listdir(data_dir)
    if files:
        st.markdown("**Active Knowledge Base Files:**")
        for f in files:
            st.markdown(f"📄 `{f}`")
    else:
        st.info("No documents uploaded yet.")

    st.markdown("---")
    
    if st.button("🚀 Rebuild Vector Index", type="primary", use_container_width=True):
        with st.spinner("Processing documents & building embeddings..."):
            chunks = load_and_split_documents(data_dir=data_dir)
            if chunks:
                emb_model = get_embedding_model()
                create_and_save_vector_store(chunks, emb_model)
                st.success(f"Indexed {len(chunks)} chunks successfully!")
            else:
                st.warning("No valid content found.")

    st.markdown("---")
    st.markdown("#### **System Telemetry**")
    
    api_key_ok = bool(os.getenv("GEMINI_API_KEY"))
    api_badge = '<span class="status-badge status-active">Connected</span>' if api_key_ok else '<span class="status-badge status-inactive">Missing Key</span>'
    st.markdown(f"**Gemini API:** {api_badge}", unsafe_allow_html=True)
    
    db_ok = os.path.exists("vectorstore/faiss_index")
    db_badge = '<span class="status-badge status-active">Active</span>' if db_ok else '<span class="status-badge status-inactive">Not Found</span>'
    st.markdown(f"**Vector Store:** {db_badge}", unsafe_allow_html=True)

# --- MAIN CHAT INTERFACE ---
st.markdown('<p class="main-title">Enterprise Intelligence & RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Secure corporate knowledge retrieval powered by Hybrid Search and Google Gemini.</p>', unsafe_allow_html=True)
st.markdown("---")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Verified Source Citations"):
                for source in message["sources"]:
                    st.markdown(f'<div class="source-card">📌 **{source}</div>', unsafe_allow_html=True)

# User Input Handling
if user_query := st.chat_input("Ask a question about your enterprise documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & generating response..."):
            vector_db_path = "vectorstore/faiss_index"
            if not os.path.exists(vector_db_path):
                response_data = {
                    "answer": "⚠️ **Vector Index Not Found:** Please upload your documents in the sidebar and click **'Rebuild Vector Index'** first.",
                    "sources": []
                }
            else:
                emb_model = get_embedding_model()
                vectorstore = load_vector_store(emb_model, vector_db_path)
                
                if vectorstore:
                    retrieved_docs = retrieve_relevant_chunks(vectorstore, user_query, top_k=4)
                    response_data = generate_rag_answer(user_query, retrieved_docs)
                else:
                    response_data = {
                        "answer": "Error loading vector store. Please rebuild the index.",
                        "sources": []
                    }

        st.markdown(response_data["answer"])
        
        if response_data["sources"]:
            with st.expander("📚 Verified Source Citations"):
                for source in response_data["sources"]:
                    st.markdown(f'<div class="source-card">📌 {source}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_data["answer"],
            "sources": response_data["sources"]
        })