class RAGPipeline:
    def __init__(self, embed_model, retriever):
        self.embed_model = embed_model
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embed_model.embed_text(query)
        return self.retriever.query(query_embedding, top_k=top_k)
