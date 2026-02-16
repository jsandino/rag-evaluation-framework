from typing import List
from openai import OpenAI
import os

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
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
