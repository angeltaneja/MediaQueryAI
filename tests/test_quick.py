"""Quick test of the document loading and chunking pipeline."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.document_loader import load_directory
from src.chunking import chunk_documents

# Load sample documents
docs = load_directory("data/sample_docs")

# Chunk them
chunks = chunk_documents(docs)

# Show sample chunk
print("\n--- Sample Chunk Metadata ---")
c = chunks[0]
print(f"  Source:   {c.metadata['source']}")
print(f"  Section:  {c.metadata['section_header']}")
print(f"  Category: {c.metadata['section_category']}")
print(f"  Tokens:   {c.metadata['estimated_tokens']}")
print(f"  Content:  {c.page_content[:150]}...")

print(f"\n--- Summary ---")
print(f"  Documents loaded: {len(docs)}")
print(f"  Total chunks:     {len(chunks)}")

# Show chunk distribution per document
sources = {}
for ch in chunks:
    s = ch.metadata["source"]
    sources[s] = sources.get(s, 0) + 1

print(f"\n--- Chunks per Document ---")
for src, count in sources.items():
    print(f"  {src}: {count} chunks")
