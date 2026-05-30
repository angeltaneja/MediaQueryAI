"""
MedQueryAI - Streamlit Application
Clinical Document RAG Engine with a premium, professional UI.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))
import config
from src.document_loader import load_document, load_directory
from src.chunking import chunk_documents
from src.vector_store import VectorStore
from src.retriever import MedicalRetriever
from src.llm_chain import MedicalQAChain


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title=f"{config.APP_TITLE} — {config.APP_SUBTITLE}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS — Premium Medical Theme
# ──────────────────────────────────────────────

st.markdown(
    """
<style>
    /* === Import Google Fonts === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* === Global Styles === */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* === Header Banner === */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0;
        font-weight: 300;
    }

    /* === Chat Messages === */
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.25);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-message-ai {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: #e0e0e0;
        padding: 1.25rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* === Source Cards === */
    .source-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .source-card:hover {
        background: rgba(102, 126, 234, 0.18);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateY(-1px);
    }
    .source-card .source-header {
        font-weight: 600;
        color: #667eea;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .source-card .source-section {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
    }
    .source-card .source-score {
        color: #4ade80;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* === Stats Cards === */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .stat-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 1.5rem;
    }

    /* === Upload Area === */
    .stFileUploader > div {
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }

    /* === Buttons === */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* === Success / Error badges === */
    .badge-success {
        background: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* === Divider === */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = 0
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0


def initialize_pipeline(provider: str = "gemini", api_key: str = ""):
    """Initialize or reinitialize the RAG pipeline components."""
    vs = VectorStore()
    retriever = MedicalRetriever(vector_store=vs)
    qa_chain = MedicalQAChain(retriever=retriever, provider=provider, api_key=api_key)

    st.session_state.vector_store = vs
    st.session_state.retriever = retriever
    st.session_state.qa_chain = qa_chain

    # Update stats
    stats = vs.get_collection_stats()
    st.session_state.total_chunks = stats["total_chunks"]
    st.session_state.indexed_docs = stats.get("unique_documents", 0)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with document management and settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # LLM Provider selection
        provider_options = {
            "gemini": "🟢 Google Gemini (FREE)",
            "groq": "🟢 Groq / Llama (FREE)",
            "claude": "🔵 Anthropic Claude (Paid)",
        }
        selected_provider = st.selectbox(
            "LLM Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=0,
            help="Gemini and Groq both have generous free tiers!",
        )

        # Dynamic API key input based on provider
        key_config = {
            "gemini": {
                "label": "Google API Key",
                "placeholder": "AIza...",
                "default": config.GOOGLE_API_KEY,
                "help": "Get free key: https://ai.google.dev",
            },
            "groq": {
                "label": "Groq API Key",
                "placeholder": "gsk_...",
                "default": config.GROQ_API_KEY,
                "help": "Get free key: https://console.groq.com",
            },
            "claude": {
                "label": "Anthropic API Key",
                "placeholder": "sk-ant-...",
                "default": config.ANTHROPIC_API_KEY,
                "help": "Get key: https://console.anthropic.com",
            },
        }

        kc = key_config[selected_provider]
        api_key = st.text_input(
            kc["label"],
            type="password",
            value=kc["default"],
            help=kc["help"],
            placeholder=kc["placeholder"],
        )

        # Show model info
        model_name = config.LLM_MODELS.get(selected_provider, "")
        st.caption(f"Model: `{model_name}`")

        # Initialize pipeline button
        if st.button("🚀 Initialize Pipeline", use_container_width=True):
            if not api_key:
                st.error(f"Please enter your API key first! {kc['help']}")
            else:
                with st.spinner("Initializing RAG pipeline..."):
                    initialize_pipeline(provider=selected_provider, api_key=api_key)
                st.success(f"Pipeline ready! Using {provider_options[selected_provider]}")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ─── Document Upload ───
        st.markdown("## 📁 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload Medical Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT medical documents to index",
        )

        if uploaded_files and st.button(
            "📥 Index Documents", use_container_width=True
        ):
            if st.session_state.vector_store is None:
                initialize_pipeline(provider=selected_provider, api_key=api_key)

            with st.spinner("Processing and indexing documents..."):
                process_uploaded_files(uploaded_files)

        # Load sample docs button
        if st.button("📚 Load Sample Documents", use_container_width=True):
            if st.session_state.vector_store is None:
                initialize_pipeline(provider=selected_provider, api_key=api_key)

            with st.spinner("Loading sample medical documents..."):
                load_sample_documents()

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ─── Collection Stats ───
        st.markdown("## 📊 Index Statistics")

        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_collection_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""<div class="stat-card">
                    <div class="stat-value">{stats.get('unique_documents', 0)}</div>
                    <div class="stat-label">Documents</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""<div class="stat-card">
                    <div class="stat-value">{stats.get('total_chunks', 0)}</div>
                    <div class="stat-label">Chunks</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Show indexed document names
            if stats.get("document_names"):
                st.markdown("**Indexed Documents:**")
                for name in stats["document_names"]:
                    st.markdown(f"  📄 `{name}`")
        else:
            st.info("Initialize the pipeline to see stats.")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ─── Retrieval Settings ───
        st.markdown("## 🔧 Retrieval Settings")
        st.session_state.top_k = st.slider(
            "Top-K Results", min_value=1, max_value=10, value=5
        )

        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.qa_chain:
                    st.session_state.qa_chain.clear_history()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Index", use_container_width=True):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear_collection()
                    st.session_state.total_chunks = 0
                    st.session_state.indexed_docs = 0
                    st.success("Index cleared!")
                    st.rerun()


# ──────────────────────────────────────────────
# Document Processing
# ──────────────────────────────────────────────

def process_uploaded_files(uploaded_files):
    """Process and index uploaded files."""
    all_chunks = []

    for uploaded_file in uploaded_files:
        # Save to temp file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Load and chunk
            docs = load_document(tmp_path)
            # Override source name with original filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
        finally:
            os.unlink(tmp_path)

    # Index chunks
    if all_chunks:
        st.session_state.vector_store.add_documents(all_chunks)
        stats = st.session_state.vector_store.get_collection_stats()
        st.session_state.total_chunks = stats["total_chunks"]
        st.session_state.indexed_docs = stats.get("unique_documents", 0)
        st.success(
            f"✅ Indexed {len(uploaded_files)} file(s) → {len(all_chunks)} chunks"
        )


def load_sample_documents():
    """Load and index the sample medical documents."""
    sample_dir = config.SAMPLE_DOCS_DIR

    if not sample_dir.exists():
        st.error(f"Sample documents directory not found: {sample_dir}")
        return

    docs = load_directory(str(sample_dir))
    if not docs:
        st.warning("No documents found in sample directory.")
        return

    chunks = chunk_documents(docs)

    if chunks:
        st.session_state.vector_store.add_documents(chunks)
        stats = st.session_state.vector_store.get_collection_stats()
        st.session_state.total_chunks = stats["total_chunks"]
        st.session_state.indexed_docs = stats.get("unique_documents", 0)
        st.success(f"✅ Loaded {len(docs)} sample doc(s) → {len(chunks)} chunks")


# ──────────────────────────────────────────────
# Main Chat Interface
# ──────────────────────────────────────────────

def render_header():
    """Render the main header banner."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{config.APP_ICON} {config.APP_TITLE}</h1>
            <p>{config.APP_SUBTITLE} — Ask questions about your medical documents with AI-powered precision</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat():
    """Render the chat interface with message history."""
    # Display existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message-user">🧑‍⚕️ {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-message-ai">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Show sources if available
            if msg.get("sources"):
                with st.expander(f"📋 Sources ({len(msg['sources'])} retrieved)", expanded=False):
                    for source in msg["sources"]:
                        st.markdown(
                            f"""<div class="source-card">
                            <div class="source-header">📄 [Source {source['index']}] {source['source']}</div>
                            <div class="source-section">Section: {source['section']} | Category: {source['section_category']}</div>
                            <div class="source-score">Relevance: {source['relevance_score']:.1%}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )


def handle_user_input():
    """Handle new user questions."""
    if prompt := st.chat_input(
        "Ask a question about your medical documents...",
        key="chat_input",
    ):
        # Check if pipeline is ready
        if st.session_state.qa_chain is None:
            st.warning(
                "⚠️ Please initialize the pipeline first (click '🚀 Initialize Pipeline' in the sidebar)"
            )
            return

        if st.session_state.total_chunks == 0:
            st.warning(
                "⚠️ No documents indexed yet. Upload documents or load sample docs from the sidebar."
            )
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.spinner("🔍 Searching documents and generating answer..."):
            result = st.session_state.qa_chain.query(
                question=prompt,
                top_k=st.session_state.get("top_k", config.TOP_K_RESULTS),
            )

        # Add AI response with sources
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )

        st.rerun()


# ──────────────────────────────────────────────
# Welcome Screen
# ──────────────────────────────────────────────

def render_welcome():
    """Show welcome message when no conversation exists."""
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 2rem;">
                <h2 style="color: rgba(255,255,255,0.8); font-weight: 300; font-size: 1.5rem;">
                    Welcome to MedQueryAI
                </h2>
                <p style="color: rgba(255,255,255,0.4); font-size: 1rem; max-width: 600px; margin: 1rem auto;">
                    Upload your medical documents and ask questions in natural language.
                    The AI will search through your documents and provide accurate, cited answers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Example queries
        st.markdown("#### 💡 Example Questions")
        example_cols = st.columns(3)

        examples = [
            "What medications is the patient currently taking?",
            "What is the recommended first-line therapy for Type 2 Diabetes?",
            "What were the discharge instructions for the pneumonia patient?",
        ]

        for col, example in zip(example_cols, examples):
            with col:
                st.markdown(
                    f"""<div class="source-card" style="cursor: default;">
                    <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">{example}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_header()
    render_welcome()
    render_chat()
    handle_user_input()


if __name__ == "__main__":
    main()
