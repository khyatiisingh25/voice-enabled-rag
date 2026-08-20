from mlx_lm import load, generate


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

MAX_TOKENS = 12

_SYSTEM_PROMPT = (
    "You are a grounded RAG answer generator. "
    "Answer ONLY using the provided context. "
    "Do not use outside knowledge. "
    "Do not guess. "
    "Give one short factual sentence. "
    "If the answer is not supported by the context, "
    "say: I do not have enough information."
)


class MLXGenerator:
    def __init__(self):
        print("Loading MLX 135M model...")
        self.model, self.tokenizer = load(MODEL_NAME)

    def _build_prompt(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer:"
                ),
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_answer(
        self,
        query: str,
        retrieved_documents: list,
    ) -> str:

        if not retrieved_documents:
            return "I do not have enough information."

        context = "\n\n".join(
            document["text"]
            for document in retrieved_documents
        )

        prompt = self._build_prompt(
            query,
            context,
        )

        answer = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        return answer.strip()
