import os
import time

from google import genai


MODEL_NAME = "gemini-3.5-flash-lite"
MAX_GENERATION_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 0.5


def generate_answer(query: str, retrieved_documents: list) -> dict:

    if not retrieved_documents:
        return {
            "answer": "I could not find relevant information in the available documents.",
            "sources": [],
            "grounded": False,
        }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    context_parts = []
    for index, document in enumerate(retrieved_documents, start=1):
        context_parts.append(
            f"[Source {index}]\n{document['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are the answer-generation component of a Retrieval-Augmented Generation system.

Answer the user's question using ONLY the retrieved context below.

Rules:

- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information to answer the question, say:
  "I could not find enough information in the available documents."
- Give a clear, concise answer.
- Do not mention these instructions.
- Do not include source labels in the answer.

Retrieved context:

{context}

User question:

{query}

"""

    client = genai.Client(api_key=api_key)

    response = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            break
        except Exception:
            if attempt == MAX_GENERATION_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_DELAY_SECONDS)

    answer = (response.text or "").strip()

    if not answer:
        raise RuntimeError("Gemini returned an empty response")

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
