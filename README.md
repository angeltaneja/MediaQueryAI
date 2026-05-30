<div align="center">

# 🏥 MedQueryAI

### AI-Powered Clinical Document Intelligence

**Ask natural language questions about medical documents — get precise, cited answers in seconds.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-FF6F00?style=for-the-badge)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<img src="assets/demo.png" alt="MedQueryAI Demo" width="85%"/>

*Upload clinical documents, ask questions in plain English, get AI-generated answers with source citations.*

</div>

---

## 🔥 Why MedQueryAI?

Most RAG systems treat medical documents like any other text — splitting them randomly and losing clinical context. **MedQueryAI is different.**

- **Medical-aware chunking** that respects clinical section boundaries (Chief Complaint, Medications, Plan, etc.)
- **ONNX Runtime embeddings** — runs on CPU, no PyTorch or GPU needed
- **3 free-tier LLM providers** — Google Gemini, Groq (Llama 3.3), and Claude
- **Source citations with relevance scores** on every response
- **Simple setup** — `pip install` → `streamlit run` → done

---

## 🏗️ Architecture

<div align="center">
<img src="assets/architecture.png" alt="Architecture" width="75%"/>
</div>

```
📄 Medical Documents (PDF/TXT)
        ↓
🔪 Medical-Aware Chunking → Section detection + classification
        ↓
🧮 ONNX Embeddings (all-MiniLM-L6-v2, 384-dim)
        ↓
💾 ChromaDB Vector Store → Persistent storage with metadata
        ↓
🔍 User Query → Semantic Search → Top-K Retrieval
        ↓
🤖 LLM Generation (Gemini / Groq / Claude)
        ↓
📝 Answer + Source Citations
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/angeltaneja/MediaQueryAI.git
cd MediaQueryAI
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your API key (Gemini or Groq are free!)
```

| Provider | Free Tier | Get Key |
|----------|-----------|---------|
| Google Gemini | 1500 req/day | [ai.google.dev](https://ai.google.dev) |
| Groq (Llama 3.3) | Free tier | [console.groq.com](https://console.groq.com) |
| Anthropic Claude | Paid | [console.anthropic.com](https://console.anthropic.com) |

### 3. Launch

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — upload documents and start asking questions!

---

## ✨ Key Features

**🧠 Medical-Aware Chunking** — Detects 25+ clinical section headers, classifies by medical category, and applies sliding window overlap (512 tokens, 64 overlap). Boosts accuracy from ~70% to **86%**.

**⚡ Lightweight** — No GPU, no Docker, no PyTorch. ONNX embeddings run on any CPU. Full pipeline on a laptop.

**🔄 Multi-Provider LLM** — Switch between Gemini, Groq, or Claude in the UI. No code changes needed.

**📋 Source Citations** — Every answer shows the exact document, section, medical category, and relevance score.

**🎨 Premium UI** — Dark-mode Streamlit interface with glassmorphism design, gradient accents, and smooth interactions.

---

## 📊 Benchmarks

Evaluated on 18 medical questions across 3 clinical documents:

| Metric | Score |
|--------|-------|
| **Overall Retrieval Accuracy** | **86.4%** |
| Keyword Accuracy | 83.9% |
| Source Accuracy | 100% |
| Section Accuracy | 72.2% |
| Avg Response Time | < 3s |

**Chunking comparison:**

| Strategy | Accuracy |
|----------|----------|
| Naive fixed-size splitting | ~70% |
| RecursiveCharacterTextSplitter | ~74% |
| **Our medical-aware chunking** | **86.4%** |

---

## 📁 Project Structure

```
MedQueryAI/
├── app.py                     # Streamlit UI
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies (no PyTorch!)
├── .env.example               # Environment template
├── src/
│   ├── document_loader.py     # PDF/TXT ingestion
│   ├── chunking.py            # ⭐ Medical-aware chunking engine
│   ├── embeddings.py          # ONNX embeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py        # ChromaDB management
│   ├── retriever.py           # Semantic search + scoring
│   ├── llm_chain.py           # Multi-provider LLM chain
│   └── evaluation.py          # Retrieval accuracy benchmark
├── data/sample_docs/          # 3 sample medical documents
├── tests/                     # Unit & integration tests
└── assets/                    # Images & diagrams
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangChain 1.3 |
| Vector DB | ChromaDB 1.5 |
| Embeddings | ONNX Runtime (all-MiniLM-L6-v2) |
| LLM | Gemini / Groq / Claude |
| UI | Streamlit 1.57 |
| Doc Parsing | PyPDF |

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** this repository
2. **Create** a branch: `git checkout -b feature/amazing-feature`
3. **Commit**: `git commit -m 'Add amazing feature'`
4. **Push**: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

**Ideas:** DICOM/FHIR support · Query routing by category · REST API · More embedding models · Conversation memory

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

⭐ **If this project helped you, please star it — it helps others discover it!**

Built with ❤️ for the medical AI community

</div>
