from rag_eval.evaluation import retrieval_accuracy, recall_at_k


def test_retrieval_metrics():
    results = [
        {
            "query": "q1",
            "ground_truth_doc_id": 1,
            "retrieved_doc_ids": [1, 2, 3],
        },
        {
            "query": "q2",
            "ground_truth_doc_id": 2,
            "retrieved_doc_ids": [3, 2, 1],
        },
        {
            "query": "q3",
            "ground_truth_doc_id": 3,
            "retrieved_doc_ids": [4, 5, 6],
        },
    ]

    accuracy = retrieval_accuracy(results)
    recall = recall_at_k(results)

    assert accuracy == 1 / 3
    assert recall == 2 / 3
