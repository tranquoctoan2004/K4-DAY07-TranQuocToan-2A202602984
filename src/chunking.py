from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Cắt SAU dấu . ! ? (lookbehind giữ nguyên dấu câu trong câu) hoặc tại newline.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip())]
        sentences = [s for s in sentences if s]

        n = self.max_sentences_per_chunk
        return [" ".join(sentences[i : i + n]) for i in range(0, len(sentences), n)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text, self.separators)

    def _hard_cut(self, text: str) -> list[str]:
        """Base case: không còn separator nào -> cắt cứng mỗi chunk_size ký tự."""
        size = max(1, self.chunk_size)
        pieces = [text[i : i + size].strip() for i in range(0, len(text), size)]
        return [p for p in pieces if p]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # 1) Đủ ngắn thì giữ nguyên.
        if len(current_text.strip()) <= self.chunk_size:
            stripped = current_text.strip()
            return [stripped] if stripped else []

        # 2) Base case: hết separator (hoặc gặp "") -> cắt cứng.
        if not remaining_separators or remaining_separators[0] == "":
            return self._hard_cut(current_text)

        sep, rest = remaining_separators[0], remaining_separators[1:]
        parts = current_text.split(sep)
        if len(parts) == 1:  # separator này không xuất hiện -> thử separator nhỏ hơn
            return self._split(current_text, rest)

        # Gắn lại separator vào cuối mỗi mảnh (trừ mảnh cuối) để không mất dấu câu/xuống dòng.
        tokens = [p + sep for p in parts[:-1]] + [parts[-1]]

        # 3) Gom các mảnh liền kề tới gần chunk_size, mảnh nào còn quá dài thì đệ quy.
        chunks: list[str] = []
        buf = ""
        for token in tokens:
            if not token.strip():
                continue
            if len((buf + token).strip()) <= self.chunk_size:
                buf += token
                continue
            if buf.strip():
                chunks.append(buf.strip())
            buf = ""
            if len(token.strip()) <= self.chunk_size:
                buf = token
            else:
                chunks.extend(self._split(token, rest))
        if buf.strip():
            chunks.append(buf.strip())
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = _dot(vec_a, vec_b) / (norm_a * norm_b)
    return max(-1.0, min(1.0, value))  # chặn sai số dấu phẩy động


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        result: dict = {}
        for name, chunker in chunkers.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result