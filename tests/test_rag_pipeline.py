from rag_eval.datasets import load_documents, load_queries
from rag_eval.embeddings import EmbeddingModel
from rag_eval.retrieval import Retriever
from rag_eval.rag import RAGPipeline
from rag_eval.evaluation import (
    run_retrieval_evaluation,
    retrieval_accuracy,
    recall_at_k,
)

# tests/conftest.py or tests/fixtures.py
import pytest
from unittest.mock import patch
from rag_eval.embeddings import EmbeddingModel
from rag_eval.datasets import load_documents, load_queries


@pytest.fixture(autouse=True)
def fake_openai_key(monkeypatch):
    """Ensure tests always have a fake OpenAI key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def mocked_rag_environment(monkeypatch):
    """
    Provide a mocked RAG environment with fake embeddings.
    """

    raw_documents = load_documents()
    queries = load_queries()

    embedding_dim = 16
    fake_embedding = [0.1] * embedding_dim

    # Add fake embeddings to documents
    documents = [{**doc, "embedding": fake_embedding} for doc in raw_documents]

    # Mock embedding methods
    monkeypatch.setattr(
        EmbeddingModel,
        "embed_text",
        lambda self, text: fake_embedding,
    )

    monkeypatch.setattr(
        EmbeddingModel,
        "embed_texts",
        lambda self, texts: [fake_embedding] * len(texts),
    )

    return documents, queries


def test_rag_pipeline_end_to_end(mocked_rag_environment):
    """Smoke test: run the full RAG pipeline on the synthetic dataset."""

    documents, queries = mocked_rag_environment

    retriever = Retriever(documents)
    embed_model = EmbeddingModel()
    rag = RAGPipeline(embed_model=embed_model, retriever=retriever)

    results = run_retrieval_evaluation(rag, queries, top_k=3)

    accuracy = retrieval_accuracy(results)
    recall = recall_at_k(results)

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= recall <= 1.0
    assert len(results) == len(queries)
