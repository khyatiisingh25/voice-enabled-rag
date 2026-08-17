class Pipeline:
    def query(self, query: str) -> dict:
        return {
            "answer": f"Mock answer for: {query}",
            "sources": [],
            "grounded": False,
        }


pipeline = Pipeline()