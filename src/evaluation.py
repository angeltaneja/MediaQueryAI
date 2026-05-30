"""
MedQueryAI - Evaluation Pipeline
Measures retrieval accuracy and answer quality
to validate the 86% accuracy claim.
"""

import src.utils  # Windows console UTF-8 fix

import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config
from src.retriever import MedicalRetriever


# ──────────────────────────────────────────────
# Ground Truth Test Set
# ──────────────────────────────────────────────

EVALUATION_QA_PAIRS = [
    {
        "question": "What medications is patient John Martinez currently taking?",
        "expected_keywords": ["lisinopril", "metformin", "atorvastatin", "aspirin", "acetaminophen"],
        "expected_section": "medications",
        "expected_source": "patient_record_001.txt",
    },
    {
        "question": "What are the patient's allergies?",
        "expected_keywords": ["penicillin", "sulfa", "rash", "hives"],
        "expected_section": "allergies",
        "expected_source": "patient_record_001.txt",
    },
    {
        "question": "What was the patient's blood pressure reading?",
        "expected_keywords": ["152/94", "mmHg"],
        "expected_section": "examination",
        "expected_source": "patient_record_001.txt",
    },
    {
        "question": "What is the troponin level and is it normal?",
        "expected_keywords": ["0.03", "troponin", "normal"],
        "expected_section": "diagnostics",
        "expected_source": "patient_record_001.txt",
    },
    {
        "question": "What is the assessment for the chest pain?",
        "expected_keywords": ["musculoskeletal", "angina", "troponin negative", "cardiac"],
        "expected_section": "assessment",
        "expected_source": "patient_record_001.txt",
    },
    {
        "question": "What is the plan for hypertension management?",
        "expected_keywords": ["lisinopril", "40 mg", "blood pressure", "low-sodium"],
        "expected_section": "plan",
        "expected_source": "patient_record_001.txt",
    },
    {
        "question": "What is the first-line therapy for type 2 diabetes?",
        "expected_keywords": ["metformin", "500 mg", "titrate"],
        "expected_section": "guidelines",
        "expected_source": "diabetes_guidelines.txt",
    },
    {
        "question": "What are the diagnostic criteria for type 2 diabetes?",
        "expected_keywords": ["fasting", "126", "HbA1c", "6.5"],
        "expected_section": "guidelines",
        "expected_source": "diabetes_guidelines.txt",
    },
    {
        "question": "What are the glycemic targets for elderly patients?",
        "expected_keywords": ["7.5", "8.0", "older adults", "comorbidities"],
        "expected_section": "guidelines",
        "expected_source": "diabetes_guidelines.txt",
    },
    {
        "question": "What SGLT2 inhibitors are recommended as second-line therapy?",
        "expected_keywords": ["empagliflozin", "dapagliflozin", "canagliflozin", "cardiovascular"],
        "expected_section": "guidelines",
        "expected_source": "diabetes_guidelines.txt",
    },
    {
        "question": "What complications should be screened for in diabetic patients?",
        "expected_keywords": ["retinopathy", "nephropathy", "neuropathy", "foot"],
        "expected_section": "guidelines",
        "expected_source": "diabetes_guidelines.txt",
    },
    {
        "question": "What was the reason for Maria Thompson's hospital admission?",
        "expected_keywords": ["pneumonia", "cough", "fever", "dyspnea"],
        "expected_section": "history",
        "expected_source": "discharge_summary_001.txt",
    },
    {
        "question": "What antibiotic was used to treat the pneumonia?",
        "expected_keywords": ["ceftriaxone", "azithromycin"],
        "expected_section": "history",
        "expected_source": "discharge_summary_001.txt",
    },
    {
        "question": "What organism was identified in the sputum culture?",
        "expected_keywords": ["streptococcus", "pneumoniae"],
        "expected_section": "history",
        "expected_source": "discharge_summary_001.txt",
    },
    {
        "question": "What are the discharge medications for the pneumonia patient?",
        "expected_keywords": ["amoxicillin", "azithromycin", "acetaminophen", "guaifenesin"],
        "expected_section": "medications",
        "expected_source": "discharge_summary_001.txt",
    },
    {
        "question": "What was the patient's oxygen saturation at discharge?",
        "expected_keywords": ["97", "room air"],
        "expected_section": "plan",
        "expected_source": "discharge_summary_001.txt",
    },
    {
        "question": "What follow-up is recommended after discharge?",
        "expected_keywords": ["1 week", "chest X-ray", "6 weeks", "pneumococcal"],
        "expected_section": "plan",
        "expected_source": "discharge_summary_001.txt",
    },
    {
        "question": "What is the patient's family history regarding cardiac disease?",
        "expected_keywords": ["father", "myocardial infarction", "72"],
        "expected_section": "history",
        "expected_source": "patient_record_001.txt",
    },
]


# ──────────────────────────────────────────────
# Evaluation Functions
# ──────────────────────────────────────────────

def evaluate_retrieval(
    retriever: MedicalRetriever,
    test_set: Optional[list[dict]] = None,
    top_k: int = 5,
) -> dict:
    """
    Evaluate retrieval accuracy against the ground truth test set.

    Metrics computed:
    - Keyword Hit Rate: % of expected keywords found in retrieved chunks
    - Source Accuracy: % of queries where correct source doc is in top-k
    - Section Accuracy: % of queries where correct section is in top-k
    - Overall Retrieval Accuracy: weighted combination

    Args:
        retriever: Initialized MedicalRetriever
        test_set: List of QA pairs (defaults to built-in set)
        top_k: Number of chunks to retrieve per query

    Returns:
        Dict with detailed metrics and per-question results
    """
    if test_set is None:
        test_set = EVALUATION_QA_PAIRS

    print(f"\n📊 Running evaluation on {len(test_set)} queries (top_k={top_k})...\n")

    results = []
    total_keyword_hits = 0
    total_keywords = 0
    source_correct = 0
    section_correct = 0

    for i, qa in enumerate(test_set, 1):
        question = qa["question"]
        expected_keywords = qa["expected_keywords"]
        expected_source = qa["expected_source"]
        expected_section = qa.get("expected_section", "")

        # Retrieve chunks
        retrieval = retriever.retrieve_with_context(query=question, top_k=top_k)
        chunks = retrieval["chunks"]
        context = retrieval["context"].lower()

        # Metric 1: Keyword hit rate
        keywords_found = []
        keywords_missed = []
        for kw in expected_keywords:
            if kw.lower() in context:
                keywords_found.append(kw)
            else:
                keywords_missed.append(kw)

        keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0
        total_keyword_hits += len(keywords_found)
        total_keywords += len(expected_keywords)

        # Metric 2: Source accuracy
        retrieved_sources = [c["metadata"].get("source", "") for c in chunks]
        source_hit = expected_source in retrieved_sources
        if source_hit:
            source_correct += 1

        # Metric 3: Section accuracy
        retrieved_sections = [c["metadata"].get("section_category", "") for c in chunks]
        section_hit = expected_section in retrieved_sections if expected_section else True
        if section_hit:
            section_correct += 1

        result = {
            "question": question,
            "keyword_score": round(keyword_score, 3),
            "keywords_found": keywords_found,
            "keywords_missed": keywords_missed,
            "source_correct": source_hit,
            "section_correct": section_hit,
            "num_chunks_retrieved": len(chunks),
            "top_relevance_score": chunks[0]["relevance_score"] if chunks else 0,
        }
        results.append(result)

        # Print progress
        status = "✅" if keyword_score >= 0.6 and source_hit else "⚠️"
        print(
            f"  {status} Q{i}: keyword={keyword_score:.0%} | "
            f"source={'✓' if source_hit else '✗'} | "
            f"section={'✓' if section_hit else '✗'} | "
            f"{question[:60]}..."
        )

    # Aggregate metrics
    n = len(test_set)
    overall_keyword_accuracy = total_keyword_hits / total_keywords if total_keywords else 0
    overall_source_accuracy = source_correct / n if n else 0
    overall_section_accuracy = section_correct / n if n else 0

    # Weighted overall score (keyword accuracy matters most)
    overall_retrieval_accuracy = (
        0.50 * overall_keyword_accuracy
        + 0.30 * overall_source_accuracy
        + 0.20 * overall_section_accuracy
    )

    summary = {
        "total_questions": n,
        "keyword_accuracy": round(overall_keyword_accuracy, 3),
        "source_accuracy": round(overall_source_accuracy, 3),
        "section_accuracy": round(overall_section_accuracy, 3),
        "overall_retrieval_accuracy": round(overall_retrieval_accuracy, 3),
        "top_k": top_k,
        "per_question_results": results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Questions evaluated:      {n}")
    print(f"  Keyword Accuracy:         {overall_keyword_accuracy:.1%}")
    print(f"  Source Accuracy:           {overall_source_accuracy:.1%}")
    print(f"  Section Accuracy:         {overall_section_accuracy:.1%}")
    print(f"  ─────────────────────────────────────")
    print(f"  Overall Retrieval Accuracy: {overall_retrieval_accuracy:.1%}")
    print(f"{'='*60}\n")

    return summary


def save_evaluation_report(summary: dict, output_path: Optional[str] = None) -> str:
    """Save evaluation results to a JSON file."""
    if output_path is None:
        output_path = str(config.BASE_DIR / "evaluation_report.json")

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📁 Report saved to: {output_path}")
    return output_path


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("🏥 MedQueryAI Evaluation Pipeline")
    print("=" * 40)

    # Initialize retriever
    retriever = MedicalRetriever()

    # Check if documents are indexed
    stats = retriever.vector_store.get_collection_stats()
    if stats["total_chunks"] == 0:
        print("\n⚠️  No documents indexed! Loading sample documents first...\n")
        from src.document_loader import load_directory
        from src.chunking import chunk_documents

        docs = load_directory(str(config.SAMPLE_DOCS_DIR))
        chunks = chunk_documents(docs)
        retriever.vector_store.add_documents(chunks)

    # Run evaluation
    results = evaluate_retrieval(retriever, top_k=5)

    # Save report
    save_evaluation_report(results)
