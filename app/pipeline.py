from app.rag.loader import load_documents
from app.rag.chunker import chunk_text
from app.rag.embeddings import (
    create_embedding_model,
    create_embeddings,
)
from app.rag.retriever import Retriever
from app.rag.validator import (
    deterministic_fallback,
    REFUSAL,
)


class RAGPipeline:

    def __init__(self):

        # -----------------------------------------------------
        # 1. LOAD DOCUMENTS
        # -----------------------------------------------------

        self.documents = load_documents()

        # -----------------------------------------------------
        # 2. LOAD EMBEDDING MODEL
        # -----------------------------------------------------

        print("Loading embedding model...")

        self.model = create_embedding_model()

        # -----------------------------------------------------
        # 3. CHUNK DOCUMENTS
        # -----------------------------------------------------

        self.chunks = []

        for document in self.documents:

            chunks = chunk_text(
                document["text"]
            )

            for chunk in chunks:

                self.chunks.append(
                    {
                        "text": chunk,
                        "source": document["source"],
                    }
                )

        # -----------------------------------------------------
        # 4. CREATE DOCUMENT EMBEDDINGS
        # -----------------------------------------------------

        embeddings = create_embeddings(
            self.model,
            [
                chunk["text"]
                for chunk in self.chunks
            ],
        )

        # -----------------------------------------------------
        # 5. BUILD FAISS RETRIEVER
        # -----------------------------------------------------

        self.retriever = Retriever(
            embeddings,
            self.chunks,
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # The MLX 135M generator is intentionally NOT loaded
        # here.
        #
        # The production latency-critical path is:
        #
        # Query
        #   ↓
        # Embedding
        #   ↓
        # FAISS
        #   ↓
        # Deterministic grounded extraction
        #
        # MLX can remain available separately for experiments
        # or optional generation, but it is not required for
        # every production query.
        # -----------------------------------------------------

    def query(self, query: str) -> dict:

        # -----------------------------------------------------
        # 1. QUERY EMBEDDING
        # -----------------------------------------------------

        query_embedding = create_embeddings(
            self.model,
            [query],
        )

        # -----------------------------------------------------
        # 2. RETRIEVAL
        # -----------------------------------------------------

        retrieved_documents = (
            self.retriever.search(
                query_embedding,
                top_k=3,
            )
        )

        # -----------------------------------------------------
        # 3. SOURCE METADATA
        # -----------------------------------------------------

        sources = [
            {
                "source": document["source"],
                "score": document["score"],
            }
            for document in retrieved_documents
        ]

        # -----------------------------------------------------
        # 4. NO RETRIEVAL
        # -----------------------------------------------------

        if not retrieved_documents:

            return {
                "answer": REFUSAL,
                "sources": [],
                "grounded": False,
                "fallback": True,
                "validation_reason": "no_retrieval",
            }

        # -----------------------------------------------------
        # 5. DETERMINISTIC GROUNDED ANSWER
        # -----------------------------------------------------

        answer = deterministic_fallback(
            query,
            retrieved_documents,
        )

        # -----------------------------------------------------
        # 6. DETERMINE GROUNDING
        # -----------------------------------------------------

        grounded = (
            answer != REFUSAL
        )

        # -----------------------------------------------------
        # 7. RETURN
        # -----------------------------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "grounded": grounded,
            "fallback": True,
            "validation_reason": (
                "deterministic_fallback"
                if grounded
                else "insufficient_context"
            ),
        }


pipeline = RAGPipeline()
