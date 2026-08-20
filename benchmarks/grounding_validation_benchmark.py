import time
import re

ANSWERS = [
    (
        "What is RAG?",
        "RAG is a retrieval-augmented generation system.",
        "A Retrieval-Augmented Generation (RAG) system combines document retrieval with a language model.",
    ),
    (
        "How does RAG work?",
        "RAG retrieves relevant document chunks and gives them to the language model.",
        "When a user asks a question, the system converts the question into an embedding and retrieves the most relevant document chunks. The retrieved information is then given to the language model to generate an answer.",
    ),
    (
        "What are embeddings?",
        "Question: What are embeddings?",
        "Each chunk is converted into a numerical vector called an embedding.",
    ),
]


def normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def validate(question, answer, context):
    # Reject empty output.
    if not answer.strip():
        return False, "empty"

    # Reject obvious question copying.
    q_words = set(normalize(question))
    a_words = set(normalize(answer))

    if q_words and len(q_words & a_words) / len(q_words) >= 0.8:
        return False, "question_copy"

    # Require at least some lexical evidence from context.
    context_words = set(normalize(context))
    overlap = len(a_words & context_words)

    if overlap < 2:
        return False, "low_context_overlap"

    return True, "pass"


def main():
    total_time = 0

    print("=== GROUNDING VALIDATION BENCHMARK ===")

    for question, answer, context in ANSWERS:
        start = time.perf_counter()

        valid, reason = validate(
            question,
            answer,
            context,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        total_time += elapsed_ms

        print(f"\nQuestion: {question}")
        print(f"Answer:   {answer}")
        print(f"Result:   {'PASS' if valid else 'REJECT'}")
        print(f"Reason:   {reason}")
        print(f"Latency:  {elapsed_ms:.4f} ms")

    print(
        f"\nTotal validation time: "
        f"{total_time:.4f} ms"
    )


if __name__ == "__main__":
    main()
