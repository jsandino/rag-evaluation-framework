from typing import List, Optional
from openai import OpenAI
import os
from pathlib import Path
import pandas as pd

from rag_eval.cache import EmbeddingCache


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        cache: Optional[EmbeddingCache] = None,
    ):
        self.model_name = model_name
        self.client = OpenAI(api_key=self._load_api_key())
        self.cache = cache

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

    def get_document_embeddings(
        self,
        documents: List[dict],
        text_field: str = "text",
        embedding_field: str = "embedding",
        regenerate: bool = False,
    ) -> List[dict]:
        """
        Get document embeddings, using cache if enabled.

        Args:
            documents: List of dicts containing at least `text_field`.
            text_field: Key in each dict with the text content.
            embedding_field: Key to store the embedding.
            cache_enabled: Whether to use caching.
        Returns:
            A new list of document dicts with embeddings added.
        """
        if not regenerate and self.cache and self.cache.exists():
            return self.cache.load()

        embedded_docs = self.generate_document_embeddings(
            documents,
            text_field,
            embedding_field,
        )

        if self.cache:
            self.cache.save(embedded_docs)

        return embedded_docs

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
