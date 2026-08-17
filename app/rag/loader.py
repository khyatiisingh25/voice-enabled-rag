from pathlib import Path


def load_documents(directory: str = "data/documents"):
    documents = []

    for file_path in Path(directory).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "text": text,
            "source": str(file_path),
        })

    return documents