"""
MedQueryAI - LLM Chain (Multi-Provider: Gemini, Groq, Claude)
Handles the generation side of RAG: takes retrieved context
and produces grounded, cited medical answers.

Supported providers:
  - Google Gemini (FREE — 1500 req/day)
  - Groq (FREE tier — Llama 3.3 70B)
  - Anthropic Claude (paid)
"""

import src.utils  # Windows console UTF-8 fix

from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.retriever import MedicalRetriever


# ──────────────────────────────────────────────
# Medical QA Prompt Template
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedQueryAI, a clinical document analysis assistant. Your role is to answer medical questions accurately based ONLY on the provided document context.

## Rules:
1. **Ground your answers** in the provided context. Do not make up information.
2. **Cite your sources** using [Source N] notation when referencing specific information.
3. **If the context doesn't contain enough information** to answer the question, say so clearly. Do not hallucinate.
4. **Use medical terminology** appropriately, but explain complex terms when helpful.
5. **Be precise and structured** in your responses. Use bullet points or numbered lists for clarity.
6. **Never provide direct medical advice**. Always note that information is from documents and should be verified by a healthcare professional.

## Important Disclaimer:
This is a document retrieval system. All answers are based on the indexed medical documents and should not be considered medical advice. Always consult a qualified healthcare professional for medical decisions."""

QA_PROMPT_TEMPLATE = """Based on the following medical document excerpts, answer the question accurately.

## Retrieved Context:
{context}

## Question:
{question}

## Instructions:
- Answer based ONLY on the context above
- Cite sources using [Source N] notation
- If the information is insufficient, state that clearly
- Be precise and well-structured"""


# ──────────────────────────────────────────────
# LLM Factory — creates the right LLM instance
# ──────────────────────────────────────────────

def create_llm(provider: str, api_key: str, model_name: Optional[str] = None):
    """
    Create an LLM instance based on the provider.

    Args:
        provider: One of "gemini", "groq", "claude"
        api_key: The API key for the chosen provider
        model_name: Optional override for model name

    Returns:
        A LangChain chat model instance
    """
    model = model_name or config.LLM_MODELS.get(provider, "")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini', 'groq', or 'claude'.")


class MedicalQAChain:
    """
    RAG chain that combines retrieved medical context with
    an LLM to generate accurate, cited answers.
    Supports Gemini (free), Groq (free), and Claude.
    """

    def __init__(
        self,
        retriever: Optional[MedicalRetriever] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the QA chain.

        Args:
            retriever: Pre-initialized MedicalRetriever
            provider: LLM provider — "gemini", "groq", or "claude"
            api_key: API key for the chosen provider
            model_name: Optional model name override
        """
        self.retriever = retriever or MedicalRetriever()
        self.provider = provider or config.LLM_PROVIDER

        # Resolve API key
        if api_key:
            resolved_key = api_key
        elif self.provider == "gemini":
            resolved_key = config.GOOGLE_API_KEY
        elif self.provider == "groq":
            resolved_key = config.GROQ_API_KEY
        elif self.provider == "claude":
            resolved_key = config.ANTHROPIC_API_KEY
        else:
            resolved_key = ""

        if not resolved_key:
            print(
                f"⚠️  No API key found for provider '{self.provider}'.\n"
                f"   Set it in .env or pass it directly.\n"
                f"   The chain will fail when trying to generate answers."
            )

        # Create LLM
        model = model_name or config.LLM_MODELS.get(self.provider, "")
        self.llm = create_llm(self.provider, resolved_key, model)

        # Conversation history for follow-up questions
        self.conversation_history: list = []

        print(f"🤖 QA Chain ready — Provider: {self.provider} | Model: {model}")

    def query(
        self,
        question: str,
        top_k: int = config.TOP_K_RESULTS,
        section_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        use_history: bool = True,
    ) -> dict:
        """
        Process a medical question through the full RAG pipeline:
        Query → Retrieve → Generate

        Args:
            question: User's medical question
            top_k: Number of chunks to retrieve
            section_filter: Optional section category filter
            source_filter: Optional source document filter
            use_history: Whether to include conversation history

        Returns:
            Dict with 'answer', 'sources', 'context', and metadata
        """
        # Step 1: Retrieve relevant context
        retrieval_result = self.retriever.retrieve_with_context(
            query=question,
            top_k=top_k,
            section_filter=section_filter,
            source_filter=source_filter,
        )

        context = retrieval_result["context"]
        sources = retrieval_result["sources"]

        # Step 2: Build prompt
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add conversation history for follow-up questions
        if use_history and self.conversation_history:
            for msg in self.conversation_history[-6:]:  # keep last 3 exchanges
                messages.append(msg)

        # Add current question with context
        user_message = QA_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        messages.append(HumanMessage(content=user_message))

        # Step 3: Generate answer with LLM
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = (
                f"❌ Error generating answer: {str(e)}\n\n"
                f"💡 Make sure your API key is set correctly.\n"
                f"   Provider: {self.provider}\n"
                f"   Get a free key:\n"
                f"   • Gemini: https://ai.google.dev\n"
                f"   • Groq: https://console.groq.com"
            )

        # Step 4: Update conversation history
        if use_history:
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=answer))

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_chunks_retrieved": retrieval_result["num_chunks"],
            "context": context,
            "provider": self.provider,
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        print("🧹 Conversation history cleared.")

    def get_retriever(self) -> MedicalRetriever:
        """Access the underlying retriever for direct search."""
        return self.retriever
