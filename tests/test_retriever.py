import numpy as np
from rag_eval.retrieval import Retriever


def test_retriever_returns_most_similar_document():
    """
    Ensure Retriever returns the document whose embedding
    is closest to the query embedding.
    """

    documents = [
        {"id": 1, "text": "doc1", "embedding": [1.0, 0.0]},
        {"id": 2, "text": "doc2", "embedding": [0.0, 1.0]},
        {"id": 3, "text": "doc3", "embedding": [0.9, 0.1]},
    ]

    retriever = Retriever(documents)

    # Query vector close to doc1/doc3 but closest to doc1
    query_embedding = np.array([1.0, 0.0], dtype="float32")

    results = retriever.query(query_embedding, top_k=1)

    assert results[0]["id"] == 1
