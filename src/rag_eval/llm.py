class MockLLM:
    """
    Minimal mock LLM implementation for testing the RAG pipeline.

    The `generate` method simply echoes part of the prompt,
    allowing the pipeline to run end-to-end without calling
    an external model.
    """

    def generate(self, prompt: str) -> str:
        return f"[MockLLM response]\n\nPrompt received:\n{prompt[:200]}..."
