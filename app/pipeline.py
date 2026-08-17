class Pipeline:
    def query(self, query: str) -> dict:
        """
        Execute the query pipeline.

        This is currently a mock implementation.
        Khyati's real RAG pipeline will be integrated here later.
        """
        return {
            "answer": f"Mock answer for: {query}",
            "sources": [],
            "grounded": False,
        }


pipeline = Pipeline()