
import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
):
    """
    Split text into approximately chunk_size characters
    while keeping sentences intact.

    The overlap is maintained using complete sentences
    rather than cutting through a sentence.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed the target,
        # finalize the current chunk.
        if (
            current_sentences
            and current_length + 1 + sentence_length
            > chunk_size
        ):
            chunks.append(
                " ".join(current_sentences)
            )

            # Keep the last sentences as overlap.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(
                current_sentences
            ):
                extra = len(previous) + (
                    1 if overlap_sentences else 0
                )

                if (
                    overlap_length + extra
                    > overlap
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous,
                )

                overlap_length += extra

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)

        if current_length:
            current_length += 1

        current_length += sentence_length

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks
