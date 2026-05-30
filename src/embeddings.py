"""
MedQueryAI - Embedding Engine
Uses ChromaDB's built-in default embedding function (based on
onnxruntime + all-MiniLM-L6-v2) — lightweight, no PyTorch needed.
Supports batch processing with progress tracking.
"""

import src.utils  # Windows console UTF-8 fix

from typing import Optional

from langchain_core.documents import Document

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config


class EmbeddingEngine:
    """
    Wrapper around ChromaDB's default embedding function.
    Uses all-MiniLM-L6-v2 via onnxruntime (no PyTorch required).
    This keeps the install size ~500MB instead of ~2.5GB.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Model name (for display; ChromaDB uses its built-in model)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._ef = None
        print(f"🧠 Embedding engine initialized with model: {self.model_name}")

    @property
    def embedding_function(self):
        """Lazy-load ChromaDB's default embedding function."""
        if self._ef is None:
            print(f"⏳ Loading embedding model (lightweight ONNX runtime)...")
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._ef = DefaultEmbeddingFunction()
            print(f"✅ Model loaded!")
        return self._ef

    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for all-MiniLM-L6-v2)."""
        return config.EMBEDDING_DIMENSION

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        result = self.embedding_function([text])
        return result[0]

    def embed_texts(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress info

        Returns:
            List of embedding vectors
        """
        print(f"🔢 Embedding {len(texts)} text(s)...")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embedding_function(batch)
            all_embeddings.extend(batch_embeddings)

            if show_progress and len(texts) > batch_size:
                done = min(i + batch_size, len(texts))
                print(f"   Progress: {done}/{len(texts)} texts embedded")

        return all_embeddings

    def embed_documents(
        self, documents: list[Document], batch_size: int = 32
    ) -> list[dict]:
        """
        Generate embeddings for a list of LangChain Documents.

        Args:
            documents: List of Document objects to embed
            batch_size: Batch size for encoding

        Returns:
            List of dicts with 'document', 'embedding', and 'id' keys
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        results = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", i)
            doc_id = f"{source}_chunk_{chunk_idx}"

            results.append(
                {"id": doc_id, "document": doc, "embedding": embedding}
            )

        print(f"✅ Generated {len(results)} embeddings (dim={self.dimension})")
        return results

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a user query.

        Args:
            query: User's search query

        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
