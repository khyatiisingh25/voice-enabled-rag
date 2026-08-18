def generate_answer(query: str, retrieved_documents: list) -> dict:
    if not retrieved_documents:
        return {
            "answer": "I could not find relevant information in the available documents.",
            "sources": [],
            "grounded": False,
        }

    best_document = retrieved_documents[0]

    answer = best_document["text"]

    sources = [
        {
            "source": document["source"],
            "score": document["score"],
        }
        for document in retrieved_documents
    ]

    return {
        "answer": answer,
        "sources": sources,
        "grounded": True,
    }