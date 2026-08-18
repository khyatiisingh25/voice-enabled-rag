import faiss
import numpy as np


class Retriever:
    def __init__(self, embeddings, documents):
        self.documents = documents
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.asarray(embeddings, dtype="float32"))

    def search(self, query_embedding, top_k=3, score_threshold=0.20):
        query_vector = np.asarray(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            score = float(score)

            if score < score_threshold:
                continue

            results.append({
                "text": self.documents[index]["text"],
                "source": self.documents[index]["source"],
                "score": score,
            })

        return results
