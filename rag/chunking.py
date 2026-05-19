import re
from typing import List


_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace to single spaces."""
    return _WS_RE.sub(" ", text or "").strip()


def chunk_text_words(
    text: str,
    *,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 40,
    min_chunk_words: int = 8,
) -> List[str]:
    """Chunk text by words with overlap.

    We use a simple word-based chunker to keep dependencies low and robust on the cluster.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    words = text.split(" ")
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")

    overlap = max(0, min(chunk_overlap_words, chunk_size_words - 1))
    stride = chunk_size_words - overlap
    chunks: List[str] = []

    for start in range(0, len(words), stride):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        if len(chunk_words) < min_chunk_words:
            continue
        chunks.append(" ".join(chunk_words).strip())
        if end >= len(words):
            break

    return chunks

