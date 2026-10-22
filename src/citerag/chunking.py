import re


def split_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of about `size` characters, overlapping by `overlap`.

    Breaks on paragraph, then sentence, then word boundaries so chunks stay readable.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut > size // 2:
                    end = start + cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
