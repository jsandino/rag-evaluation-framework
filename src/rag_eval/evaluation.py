from typing import List, Dict, Any
from rag_eval.rag import RAGPipeline

"""
Evaluation utilities for retrieval-based RAG systems.

This module provides simple metrics for measuring retrieval quality
such as accuracy and recall@k.
"""


def retrieval_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Compute retrieval accuracy.

    Accuracy is defined as the proportion of queries for which the
    top retrieved document matches the ground truth document.

    Parameters
    ----------
    results : list[dict]
        Retrieval results containing:
        - retrieved_doc_ids (list)
        - ground_truth_doc_id

    Returns
    -------
    float
        Accuracy score between 0 and 1.
    """
    correct = 0

    for r in results:
        top_doc_id = r["retrieved_doc_ids"][0]

        if top_doc_id == r["ground_truth_doc_id"]:
            correct += 1

    return correct / len(results)


def recall_at_k(results: List[Dict[str, Any]]) -> float:
    """
    Compute Recall@k.

    Recall@k measures whether the correct document appears anywhere
    within the top-k retrieved results.

    Parameters
    ----------
    results : list[dict]
        Retrieval results containing:
        - retrieved_doc_ids (list)
        - ground_truth_doc_id

    Returns
    -------
    float
        Recall@k score between 0 and 1.
    """
    correct = 0

    for r in results:
        if r["ground_truth_doc_id"] in r["retrieved_doc_ids"]:
            correct += 1

    return correct / len(results)


def run_retrieval_evaluation(
    rag: RAGPipeline, queries: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """
    Run retrieval for a list of evaluation queries.

    Parameters
    ----------
    rag : RAGPipeline
        Pipeline used to perform retrieval.
    queries : List[Dict[str, Any]]
        List of query dictionaries containing:
        - "query": str
        - "ground_truth_doc_id": int
    top_k : int
        Number of documents to retrieve.

    Returns
    -------
    List[Dict[str, Any]]
        Standardized retrieval results containing:
        - "query": str
        - "ground_truth_doc_id": int
        - "retrieved_doc_ids": List[int]
    """
    results: List[Dict[str, Any]] = []

    for q in queries:
        retrieved_docs = rag.retrieve(q["query"], top_k=top_k)

        retrieved_ids = [doc["id"] for doc in retrieved_docs]

        results.append(
            {
                "query": q["query"],
                "ground_truth_doc_id": q["ground_truth_doc_id"],
                "retrieved_doc_ids": retrieved_ids,
            }
        )

    return results
