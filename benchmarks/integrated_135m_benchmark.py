import re
import statistics
import time

from mlx_lm import load, generate

from app.pipeline import pipeline
from app.rag.embeddings import create_embeddings


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

QUERIES = [
    "What is RAG?",
    "How does RAG work?",
    "What are embeddings?",
    "What does the system retrieve?",
    "What happens after retrieval?",
]


def percentile(values, p):
    values = sorted(values)

    index = max(
        0,
        min(
            len(values) - 1,
            int((p / 100) * len(values)) - 1,
        ),
    )

    return values[index]


def normalize(text):
    return re.sub(
        r"[^a-z0-9\s]",
        "",
        text.lower(),
    ).split()


def validate_answer(question, answer, context):
    answer = answer.strip()

    if not answer:
        return False, "empty"

    question_words = set(normalize(question))
    answer_words = set(normalize(answer))

    if question_words:
        question_overlap = (
            len(question_words & answer_words)
            / len(question_words)
        )

        if question_overlap >= 0.8:
            return False, "question_copy"

    context_words = set(normalize(context))
    meaningful_words = {
        word
        for word in answer_words
        if len(word) >= 4
    }

    overlap = len(
        meaningful_words & context_words
    )

    if overlap < 2:
        return False, "low_context_overlap"

    return True, "pass"


def build_prompt(tokenizer, question, context):
    system_prompt = (
        "You are a grounded RAG answer generator. "
        "Answer ONLY from the provided context. "
        "Do not use outside knowledge. "
        "Do not guess. "
        "Give one complete factual sentence. "
        "If the answer is not supported by the context, "
        "say: I do not have enough information."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    print("Loading MLX model...")

    model, tokenizer = load(MODEL_NAME)

    # Warm-up
    warmup_prompt = build_prompt(
        tokenizer,
        "What is RAG?",
        "A RAG system combines document retrieval with a language model.",
    )

    generate(
        model,
        tokenizer,
        prompt=warmup_prompt,
        max_tokens=12,
        verbose=False,
    )

    total_times = []
    embedding_times = []
    retrieval_times = []
    generation_times = []
    validation_times = []

    print("\n=== INTEGRATED 135M RAG BENCHMARK ===")

    for query in QUERIES:
        total_start = time.perf_counter()

        # Embedding
        start = time.perf_counter()

        query_embedding = create_embeddings(
            pipeline.model,
            [query],
        )

        embedding_ms = (
            time.perf_counter() - start
        ) * 1000

        # FAISS retrieval
        start = time.perf_counter()

        results = pipeline.retriever.search(
            query_embedding,
            top_k=3,
        )

        retrieval_ms = (
            time.perf_counter() - start
        ) * 1000

        context = "\n\n".join(
            doc["text"]
            for doc in results
        )

        if not context:
            context = (
                "No relevant information was retrieved."
            )

        # MLX generation
        prompt = build_prompt(
            tokenizer,
            query,
            context,
        )

        start = time.perf_counter()

        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=12,
            verbose=False,
        ).strip()

        generation_ms = (
            time.perf_counter() - start
        ) * 1000

        # Validation
        start = time.perf_counter()

        valid, reason = validate_answer(
            query,
            answer,
            context,
        )

        validation_ms = (
            time.perf_counter() - start
        ) * 1000

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        embedding_times.append(embedding_ms)
        retrieval_times.append(retrieval_ms)
        generation_times.append(generation_ms)
        validation_times.append(validation_ms)
        total_times.append(total_ms)

        print(f"\n--- {query} ---")
        print(f"Embedding : {embedding_ms:.2f} ms")
        print(f"FAISS     : {retrieval_ms:.2f} ms")
        print(f"Generation: {generation_ms:.2f} ms")
        print(f"Validation: {validation_ms:.4f} ms")
        print(f"TOTAL     : {total_ms:.2f} ms")
        print(f"Validation result: {valid} ({reason})")
        print(f"Answer: {answer}")

    print("\n=== RESULTS ===")

    print(
        f"Embedding P50: "
        f"{statistics.median(embedding_times):.2f} ms"
    )

    print(
        f"FAISS P50: "
        f"{statistics.median(retrieval_times):.2f} ms"
    )

    print(
        f"Generation P50: "
        f"{statistics.median(generation_times):.2f} ms"
    )

    print(
        f"Validation P50: "
        f"{statistics.median(validation_times):.4f} ms"
    )

    print(
        f"TOTAL P50: "
        f"{statistics.median(total_times):.2f} ms"
    )

    print(
        f"TOTAL P70: "
        f"{percentile(total_times, 70):.2f} ms"
    )

    print(
        f"TOTAL P100: "
        f"{max(total_times):.2f} ms"
    )


if __name__ == "__main__":
    main()
