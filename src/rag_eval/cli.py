import argparse

from dotenv import load_dotenv

from rag_eval.datasets import load_documents, load_queries
from rag_eval.embeddings import EmbeddingModel
from rag_eval.retrieval import Retriever
from rag_eval.rag import RAGPipeline
from rag_eval.evaluation import (
    run_retrieval_evaluation,
    retrieval_accuracy,
    recall_at_k,
)

load_dotenv()


def evaluate(top_k: int = 3) -> None:
    """Run retrieval evaluation on the dataset."""

    documents = load_documents()
    queries = load_queries()

    embed_model = EmbeddingModel()
    documents = embed_model.generate_document_embeddings(documents)

    retriever = Retriever(documents)
    rag = RAGPipeline(embed_model=embed_model, retriever=retriever)

    results = run_retrieval_evaluation(rag=rag, queries=queries, top_k=top_k)

    accuracy = retrieval_accuracy(results)
    recall = recall_at_k(results)

    print("Evaluation Results")
    print("------------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall@{top_k}: {recall:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation CLI")

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve",
    )

    args = parser.parse_args()

    evaluate(top_k=args.top_k)


if __name__ == "__main__":
    main()
