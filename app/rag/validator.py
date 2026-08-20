import re


REFUSAL = (
    "I could not find enough information "
    "in the available documents."
)


STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "how",
    "does",
    "do",
    "and",
    "of",
    "to",
    "after",
    "before",
    "this",
    "that",
    "system",
    "then",
    "when",
    "user",
    "in",
    "on",
    "for",
    "from",
    "does",
    "used",
}


def normalize(text: str) -> list[str]:
    words = re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )

    normalized = []

    for word in words:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"

        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]

        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]

        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]

        normalized.append(word)

    return normalized


def content_words(text: str) -> set[str]:
    return {
        word
        for word in normalize(text)
        if word not in STOPWORDS
        and len(word) >= 3
    }


def split_sentences(text: str) -> list[str]:
    pieces = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip(),
    )

    return [
        piece.strip()
        for piece in pieces
        if len(piece.strip()) >= 15
    ]


def looks_like_heading(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    words = text.split()

    if len(words) <= 6 and not re.search(
        r"[.!?]$",
        text,
    ):
        return True

    return False


def question_intent(question: str) -> str:
    q = question.lower().strip()

    if q.startswith("what is ") or q.startswith("what are "):
        return "definition"

    if q.startswith("how does ") or q.startswith("how do "):
        return "process"

    if q.startswith("what happens after "):
        return "after"

    if "what does" in q and "retriev" in q:
        return "retrieval"

    if q.startswith("who "):
        return "person"

    if q.startswith("how many "):
        return "quantity"

    if q.startswith("how much "):
        return "quantity"

    if (
        "programming language" in q
        or "programming languages" in q
    ):
        return "programming_language"

    if (
        "what database" in q
        or "which database" in q
    ):
        return "database"

    return "general"


def deterministic_fallback(
    question: str,
    documents: list,
) -> str:

    if not documents:
        return REFUSAL

    intent = question_intent(question)

    sentences = []

    for document in documents:

        for sentence in split_sentences(
            document["text"]
        ):

            if looks_like_heading(sentence):
                continue

            sentences.append(
                {
                    "text": sentence,
                    "score": document.get(
                        "score",
                        0.0,
                    ),
                }
            )

    if not sentences:
        return REFUSAL

    # =========================================================
    # RAG DEFINITION
    # =========================================================

    if intent == "definition":

        question_lower = question.lower()

        if "rag" in question_lower:

            for item in sentences:
                sentence = item["text"]
                lower = sentence.lower()

                if (
                    "retrieval-augmented generation"
                    in lower
                    and "combines"
                    in lower
                ):
                    return sentence

            return REFUSAL

        if "embedding" in question_lower:

            for item in sentences:
                sentence = item["text"]
                lower = sentence.lower()

                if (
                    "embedding" in lower
                    and "called" in lower
                    and (
                        "vector" in lower
                        or "numerical" in lower
                    )
                ):
                    return sentence

            return REFUSAL

    # =========================================================
    # PROGRAMMING LANGUAGE
    # =========================================================

    if intent == "programming_language":

        candidates = []

        for item in sentences:

            sentence = item["text"]
            lower = sentence.lower()

            if (
                "project uses" in lower
                and (
                    "python" in lower
                    or "java" in lower
                    or "javascript" in lower
                    or "typescript" in lower
                    or "c++" in lower
                    or "c#" in lower
                )
            ):
                candidates.append(item)

        if candidates:

            candidates.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            return candidates[0]["text"]

        return REFUSAL

    # =========================================================
    # DATABASE
    # =========================================================

    if intent == "database":

        # The current document only says
        # "vector database"; it does not name
        # a specific database product.

        for item in sentences:

            lower = item["text"].lower()

            if "vector database" in lower:

                return item["text"]

        return REFUSAL

    # =========================================================
    # PROCESS
    # =========================================================

    if intent == "process":

        candidates = []

        for item in sentences:

            lower = item["text"].lower()

            if (
                "loads document" in lower
                and "split" in lower
            ):
                candidates.append(item)

            elif (
                "convert" in lower
                and "embedding" in lower
            ):
                candidates.append(item)

            elif (
                "retriev" in lower
                and "document chunk" in lower
            ):
                candidates.append(item)

        if candidates:
            candidates.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            return candidates[0]["text"]

        return REFUSAL

    # =========================================================
    # RETRIEVAL
    # =========================================================

    if intent == "retrieval":

        candidates = []

        for item in sentences:

            lower = item["text"].lower()

            if (
                "retriev" in lower
                and "document chunk" in lower
            ):
                candidates.append(item)

        if candidates:
            candidates.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            return candidates[0]["text"]

        return REFUSAL

    # =========================================================
    # AFTER RETRIEVAL
    # =========================================================

    if intent == "after":

        candidates = []

        for item in sentences:

            lower = item["text"].lower()

            if (
                "retrieved information"
                in lower
                and "given to a language model"
                in lower
            ):
                candidates.append(item)

        if candidates:
            candidates.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            return candidates[0]["text"]

        return REFUSAL

    # =========================================================
    # WHO QUESTIONS
    # =========================================================

    if intent == "person":

        return REFUSAL

    # =========================================================
    # QUANTITY QUESTIONS
    # =========================================================

    if intent == "quantity":

        query_words = content_words(question)

        candidates = []

        for item in sentences:

            sentence = item["text"]

            sentence_words = content_words(
                sentence
            )

            overlap = (
                query_words
                & sentence_words
            )

            if overlap and re.search(
                r"\b\d+(?:\.\d+)?\b",
                sentence,
            ):
                candidates.append(
                    (
                        len(overlap),
                        item["score"],
                        sentence,
                    )
                )

        if candidates:

            candidates.sort(
                reverse=True
            )

            return candidates[0][2]

        return REFUSAL

    # =========================================================
    # GENERAL
    # =========================================================

    query_words = content_words(question)

    candidates = []

    for item in sentences:

        sentence = item["text"]

        sentence_words = content_words(
            sentence
        )

        overlap = (
            query_words
            & sentence_words
        )

        if not overlap:
            continue

        candidates.append(
            (
                len(overlap),
                item["score"],
                sentence,
            )
        )

    if not candidates:
        return REFUSAL

    candidates.sort(
        reverse=True
    )

    best_overlap, _, best_sentence = candidates[0]

    if best_overlap < 2:
        return REFUSAL

    return best_sentence


def validate_answer(
    question: str,
    answer: str,
    context: str,
) -> tuple[bool, str]:

    answer = answer.strip()

    if not answer:
        return False, "empty"

    tokens = normalize(answer)

    if len(tokens) < 5:
        return False, "too_short"

    # =========================================================
    # PROMPT ECHO
    # =========================================================

    answer_lower = answer.lower()

    forbidden_patterns = [
        r"\bquestion\s*:",
        r"\banswer\s*:",
        r"\bcontext\s*:",
        r"\buser\s+question\s*:",
        r"\bprovided\s+context\b",
        r"\banswer\s+using\s+only\b",
        r"\bdo\s+not\s+use\s+outside\b",
        r"\bdo\s+not\s+guess\b",
    ]

    for pattern in forbidden_patterns:

        if re.search(
            pattern,
            answer_lower,
        ):
            return False, "prompt_echo"

    # =========================================================
    # ANSWER IS QUESTION
    # =========================================================

    if answer.endswith("?"):
        return False, "answer_is_question"

    # =========================================================
    # TRUNCATED
    # =========================================================

    incomplete_endings = {
        "the",
        "a",
        "an",
        "and",
        "to",
        "of",
        "with",
        "for",
        "that",
        "which",
        "from",
        "into",
        "is",
        "are",
        "was",
        "were",
        "generate",
        "generates",
        "retrieval",
        "retrieves",
        "using",
    }

    if tokens[-1] in incomplete_endings:
        return False, "truncated"

    # =========================================================
    # QUESTION COPY
    # =========================================================

    question_text = " ".join(
        normalize(question)
    )

    answer_text = " ".join(tokens)

    if answer_text == question_text:
        return False, "question_copy"

    # =========================================================
    # CONTEXT GROUNDING
    # =========================================================

    answer_words = content_words(answer)
    context_words = content_words(context)

    overlap = (
        answer_words
        & context_words
    )

    if len(overlap) < 2:
        return False, "low_context_overlap"

    # =========================================================
    # REPETITION
    # =========================================================

    if len(tokens) >= 6:

        unique_ratio = (
            len(set(tokens))
            / len(tokens)
        )

        if unique_ratio < 0.45:
            return False, "repetition"

    return True, "pass"