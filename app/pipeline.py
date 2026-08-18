from app.rag.loader import load_documents
from app.rag.chunker import chunk_text
from app.rag.embeddings import create_embedding_model, create_embeddings
from app.rag.retriever import Retriever
from app.rag.generator import generate_answer


class RAGPipeline:
    def __init__(self):
        self.documents = load_documents()
        self.model = create_embedding_model()

        self.chunks = []

        for document in self.documents:
            chunks = chunk_text(document["text"])

            for chunk in chunks:
                self.chunks.append({
                    "text": chunk,
                    "source": document["source"],
                })

        embeddings = create_embeddings(
            self.model,
            [chunk["text"] for chunk in self.chunks]
        )

        self.retriever = Retriever(
            embeddings,
            self.chunks
        )

    def query(self, query: str) -> dict:
        query_embedding = create_embeddings(
            self.model,
            [query]
        )

        retrieved_documents = self.retriever.search(
            query_embedding,
            top_k=3
        )

        return generate_answer(
            query,
            retrieved_documents
        )
pipeline = RAGPipeline()
