from pathlib import Path
from typing import Dict, List

import pandas as pd


class EmbeddingCache:
    """Simple cache for storing and retrieving document embeddings using Parquet files."""

    EMBEDDINGS_FILE = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "embeddings.parquet"
    )

    def __init__(self, file_path: Path = EMBEDDINGS_FILE):
        self.file_path = file_path

    def exists(self) -> bool:
        """
        Check if the cache file exists.
        """
        return self.file_path.exists()

    def load(self) -> List[Dict]:
        """
        Load embeddings from a Parquet file and return as list of dicts.

        Args:
            file_path: Path to parquet file.

        Returns:
            List of dicts with embeddings.
        """
        df = pd.read_parquet(self.file_path)
        return df.to_dict(orient="records")

    def save(self, documents: List[Dict]) -> None:
        """
        Save documents (with embeddings) to a Parquet file.

        Args:
            documents: List of dicts with 'embedding' key.
            file_path: Path to write the parquet file.
        """
        df = pd.DataFrame(documents)
        df.to_parquet(self.file_path, index=False)

    def __str__(self):
        return f"EmbeddingCache(file_path={self.file_path})"
