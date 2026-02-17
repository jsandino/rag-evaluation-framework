from typing import List
from openai import OpenAI
import os
from pathlib import Path
import pandas as pd


class EmbeddingModel:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = OpenAI(api_key=self._load_api_key())

    @staticmethod
    def _load_api_key() -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Set it in your environment or provide a .env file."
            )
        return api_key

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text by calling embed_texts under the hood.
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        """
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def generate_document_embeddings(
        self,
        documents: List[dict],
        text_field: str = "text",
        embedding_field: str = "embedding",
    ) -> List[dict]:
        """
        Generates embeddings for each document and adds them under `embedding_field`.

        Args:
            documents: List of dicts containing at least `text_field`.
            text_field: Key in each dict with the text content.
            embedding_field: Key to store the embedding.

        Returns:
            A new list of document dicts with embeddings added.
        """
        result = []  # Copy of documents with embeddings added

        # Extract document texts
        texts = [doc[text_field] for doc in documents]

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Attach embeddings to the documents (for inline inspection)
        for doc, emb in zip(documents, embeddings):
            new_doc = doc.copy()  # create a shallow copy
            new_doc[embedding_field] = emb
            result.append(new_doc)
        return result

    def save_embeddings(self, documents: list[dict], file_path: Path) -> None:
        """
        Save documents (with embeddings) to a Parquet file.

        Args:
            documents: List of dicts with 'embedding' key.
            file_path: Path to write the parquet file.
        """
        df = pd.DataFrame(documents)
        df.to_parquet(file_path, index=False)

    def load_embeddings(self, file_path: Path) -> list[dict]:
        """
        Load embeddings from a Parquet file and return as list of dicts.

        Args:
            file_path: Path to parquet file.

        Returns:
            List of dicts with embeddings.
        """
        df = pd.read_parquet(file_path)
        return df.to_dict(orient="records")
