"""
MedQueryAI - ChromaDB Vector Store Management
Handles persistent storage, retrieval, and management of
document embeddings in ChromaDB.
"""

import src.utils  # Windows console UTF-8 fix

import hashlib
from typing import Optional

import chromadb
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.embeddings import EmbeddingEngine


class VectorStore:
    """
    ChromaDB-backed vector store for medical document chunks.
    Supports persistent storage, incremental updates, and
    metadata-filtered search.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory for persistent storage
            embedding_engine: Pre-initialized EmbeddingEngine instance
        """
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.persist_dir = persist_dir or str(config.CHROMA_PERSIST_DIR)
        self.embedding_engine = embedding_engine or EmbeddingEngine()

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create the collection (cosine distance for normalized similarity scores)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "MedQueryAI medical document embeddings",
                "hnsw:space": "cosine",
            },
        )

        print(
            f"📦 Vector store ready: '{self.collection_name}' "
            f"({self.collection.count()} existing documents)"
        )

    def _generate_doc_id(self, text: str, metadata: dict) -> str:
        """Generate a deterministic ID for a document chunk."""
        source = metadata.get("source", "unknown")
        chunk_idx = metadata.get("chunk_index", 0)
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{source}_chunk{chunk_idx}_{content_hash}"

    def _serialize_metadata(self, metadata: dict) -> dict:
        """
        Ensure all metadata values are ChromaDB-compatible types
        (str, int, float, bool). Convert others to strings.
        """
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif value is None:
                serialized[key] = ""
            else:
                serialized[key] = str(value)
        return serialized

    def add_documents(
        self, documents: list[Document], batch_size: int = 32
    ) -> int:
        """
        Embed and store documents in ChromaDB.

        Args:
            documents: List of chunked Document objects
            batch_size: Batch size for embedding generation

        Returns:
            Number of documents added
        """
        if not documents:
            print("⚠️  No documents to add.")
            return 0

        print(f"\n📥 Adding {len(documents)} chunks to vector store...")

        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_engine.embed_texts(texts, batch_size=batch_size)

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        for doc in documents:
            doc_id = self._generate_doc_id(doc.page_content, doc.metadata)
            ids.append(doc_id)
            metadatas.append(self._serialize_metadata(doc.metadata))

        # Upsert in batches (ChromaDB handles deduplication by ID)
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.upsert(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            total_added += batch_end - i

        print(
            f"✅ Stored {total_added} chunks. "
            f"Collection total: {self.collection.count()}"
        )
        return total_added

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RESULTS,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform semantic search over the vector store.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            where: Optional ChromaDB metadata filter

        Returns:
            List of result dicts with 'content', 'metadata', 'distance', 'id'
        """
        # Embed the query
        query_embedding = self.embedding_engine.embed_query(query)

        # Search ChromaDB
        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            search_kwargs["where"] = where

        results = self.collection.query(**search_kwargs)

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1 - results["distances"][0][i],  # cosine distance → similarity [0, 1]
                }
            )

        return formatted_results

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        count = self.collection.count()

        stats = {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": self.persist_dir,
        }

        # Get unique source documents
        if count > 0:
            all_docs = self.collection.get(include=["metadatas"])
            sources = set()
            sections = set()
            for meta in all_docs["metadatas"]:
                sources.add(meta.get("source", "unknown"))
                sections.add(meta.get("section_header", "unknown"))
            stats["unique_documents"] = len(sources)
            stats["document_names"] = sorted(sources)
            stats["unique_sections"] = len(sections)

        return stats

    def delete_by_source(self, source_name: str) -> int:
        """
        Delete all chunks from a specific source document.

        Args:
            source_name: The filename of the source to delete

        Returns:
            Number of chunks deleted
        """
        # Get IDs of chunks from this source
        results = self.collection.get(
            where={"source": source_name}, include=["metadatas"]
        )

        if not results["ids"]:
            print(f"⚠️  No chunks found for source: {source_name}")
            return 0

        count = len(results["ids"])
        self.collection.delete(ids=results["ids"])
        print(f"🗑️  Deleted {count} chunks from '{source_name}'")
        return count

    def clear_collection(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "MedQueryAI medical document embeddings"},
        )
        print(f"🗑️  Collection '{self.collection_name}' cleared.")
