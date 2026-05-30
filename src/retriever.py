"""
MedQueryAI - Retriever
Handles semantic search, re-ranking, and metadata-filtered
retrieval from the vector store.
"""

from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.vector_store import VectorStore


class MedicalRetriever:
    """
    Retrieval engine that fetches relevant medical document chunks
    for a given query, with support for re-ranking and filtering.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the retriever.

        Args:
            vector_store: Pre-initialized VectorStore instance.
                         Creates a new one if not provided.
        """
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K_RESULTS,
        section_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        min_score: float = config.SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: Natural language question
            top_k: Maximum number of chunks to return
            section_filter: Filter by section category (e.g., 'medications', 'assessment')
            source_filter: Filter by source document name
            min_score: Minimum relevance score threshold

        Returns:
            List of relevant chunks with content, metadata, and scores
        """
        # Build metadata filter
        where_filter = None
        if section_filter or source_filter:
            conditions = {}
            if section_filter:
                conditions["section_category"] = section_filter
            if source_filter:
                conditions["source"] = source_filter
            where_filter = conditions

        # Search the vector store
        raw_results = self.vector_store.search(
            query=query, top_k=top_k * 2, where=where_filter  # over-fetch for re-ranking
        )

        # Filter by minimum relevance score
        filtered = [r for r in raw_results if r["relevance_score"] >= min_score]

        # Re-rank: sort by relevance score (highest first)
        filtered.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Trim to top_k
        results = filtered[:top_k]

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = config.TOP_K_RESULTS,
        section_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> dict:
        """
        Retrieve chunks and format them into a context string
        suitable for LLM input.

        Args:
            query: Natural language question
            top_k: Number of chunks to retrieve
            section_filter: Optional section category filter
            source_filter: Optional source document filter

        Returns:
            Dict with 'context' string, 'sources' list, and 'chunks' list
        """
        chunks = self.retrieve(
            query=query,
            top_k=top_k,
            section_filter=section_filter,
            source_filter=source_filter,
        )

        if not chunks:
            return {
                "context": "No relevant information found in the indexed documents.",
                "sources": [],
                "chunks": [],
                "num_chunks": 0,
            }

        # Build context string with source citations
        context_parts = []
        sources = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            section = chunk["metadata"].get("section_header", "General")
            score = chunk["relevance_score"]

            context_parts.append(
                f"[Source {i}: {source} | Section: {section} | Relevance: {score:.2f}]\n"
                f"{chunk['content']}"
            )

            source_info = {
                "index": i,
                "source": source,
                "section": section,
                "relevance_score": round(score, 3),
                "section_category": chunk["metadata"].get("section_category", "general"),
            }
            sources.append(source_info)

        context = "\n\n---\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sources,
            "chunks": chunks,
            "num_chunks": len(chunks),
        }

    def get_available_filters(self) -> dict:
        """
        Get available metadata values for filtering.
        Useful for UI dropdowns.
        """
        stats = self.vector_store.get_collection_stats()
        return {
            "documents": stats.get("document_names", []),
            "total_chunks": stats.get("total_chunks", 0),
        }
