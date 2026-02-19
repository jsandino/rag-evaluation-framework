class RAGPipeline:
    def __init__(self, embed_model, retriever, llm=None):
        """
        Orchestrates embedding, retrieval, and (later) generation.

        Parameters
        ----------
        embed_model : EmbeddingModel
            Responsible for embedding queries.
        retriever : Retriever
            Responsible for retrieving top-k documents.
        llm : object
            Any object exposing a `generate(prompt: str) -> str` method.
        """
        self.embed_model = embed_model
        self.retriever = retriever
        self.llm = llm

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve relevant documents for a given query."""
        query_embedding = self.embed_model.embed_text(query)
        return self.retriever.query(query_embedding, top_k=top_k)

    def generate(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant documents and generate an answer using the LLM.
        """
        if self.llm is None:
            raise ValueError(
                "No LLM provided. Pass an LLM to RAGPipeline to enable generation."
            )

        # Step 1: Retrieve documents
        retrieved_docs = self.retrieve(query, top_k=top_k)

        # Step 2: Build context string
        context = "\n\n".join(doc["text"] for doc in retrieved_docs)

        # Step 3: Construct simple prompt
        prompt = (
            "Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )

        # Step 4: Call LLM
        return self.llm.generate(prompt)
