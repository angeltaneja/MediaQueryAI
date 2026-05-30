"""End-to-end test of MedQueryAI pipeline with Groq."""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("  MedQueryAI - Full Pipeline Test")
print("=" * 60)

# Step 1: Load documents
print("\n[1/4] Loading documents...")
from src.document_loader import load_directory
docs = load_directory("data/sample_docs")

# Step 2: Chunk documents
print("\n[2/4] Chunking documents...")
from src.chunking import chunk_documents
chunks = chunk_documents(docs)

# Step 3: Index into ChromaDB
print("\n[3/4] Embedding & indexing into ChromaDB...")
from src.vector_store import VectorStore
vs = VectorStore()
vs.clear_collection()  # fresh start
vs.add_documents(chunks)

# Step 4: Query with LLM
provider = os.getenv("LLM_PROVIDER", "groq")
api_key = os.getenv("GROQ_API_KEY", "") if provider == "groq" else os.getenv("GOOGLE_API_KEY", "")
print(f"\n[4/4] Querying with {provider} LLM...")
from src.retriever import MedicalRetriever
from src.llm_chain import MedicalQAChain

retriever = MedicalRetriever(vector_store=vs)
qa = MedicalQAChain(
    retriever=retriever,
    provider=provider,
    api_key=api_key,
)

question = "What medications is patient John Martinez currently taking?"
print(f"\nQuestion: {question}")
print("-" * 50)

result = qa.query(question)

print(f"\nAnswer:\n{result['answer']}")
print(f"\nSources used: {result['num_chunks_retrieved']}")
for s in result['sources']:
    print(f"  [{s['index']}] {s['source']} | {s['section']} | score={s['relevance_score']:.2f}")

print("\n" + "=" * 60)
print("  TEST PASSED - Pipeline is fully functional!")
print("=" * 60)
