import faiss
import numpy as np


class Retriever:
    """
    FAISS-based retrieval class for vector embeddings.

    This class builds a FAISS index from a set of documents with precomputed embeddings
    and provides a simple API to retrieve the top-k most similar documents for a given query embedding.

    Attributes
    ----------
    documents : list[dict]
        List of documents, each a dict containing keys:
        - 'id' : unique identifier
        - 'text' : document text
        - 'embedding' : vector embedding (list or np.ndarray)
    index : faiss.Index
        FAISS index constructed from document embeddings (L2 distance).

    Methods
    -------
    query(embedding: list, top_k: int = 1) -> list[dict]:
        Retrieve the top-k most similar documents for a single query embedding.
    """

    def __init__(self, documents: list):
        """
        documents: list of dicts with keys 'id', 'text', 'embedding'
        """
        self.documents = documents
        self.index = None
        self._build_index()

    def _build_index(self):
        # Convert embeddings to numpy matrix
        embeddings = [doc["embedding"] for doc in self.documents]
        embedding_matrix = np.vstack(embeddings).astype("float32")

        # Initialize FAISS index (L2 distance)
        embedding_dim = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Add embeddings to index
        self.index.add(embedding_matrix)

    def query(self, embedding: list, top_k: int = 1):
        """
        Retrieve top_k documents for a single query embedding.

        embedding: list or np.ndarray representing a single query
        Returns: list of document dicts
        """
        # Convert to 2D array for FAISS
        embedding_matrix = np.array([embedding], dtype=np.float32)
        D, I = self.index.search(embedding_matrix, top_k)
        return [self.documents[i] for i in I[0]]
