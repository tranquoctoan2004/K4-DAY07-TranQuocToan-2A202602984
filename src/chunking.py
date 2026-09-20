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


class HeadingChunker:
    """
    Split Markdown text into chunks aligned to headings (## Điều ..., ### ...).

    Rules:
        - Mỗi heading (bất kỳ cấp độ #, ##, ###...) mở ra một section mới.
        - Nếu section đủ ngắn (<= chunk_size), giữ nguyên làm 1 chunk.
        - Nếu section quá dài, hạ xuống RecursiveChunker để chia nhỏ,
          rồi gắn lại tiêu đề heading vào đầu mỗi mảnh con để không mất ngữ cảnh.
        - Phần text đứng trước heading đầu tiên (nếu có, không tính H1 title)
          được coi là một section riêng (thường là đoạn mở đầu / lời giới thiệu).
    """

    HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+.*$")

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            # Không có heading nào -> coi cả văn bản là 1 section, hạ xuống recursive nếu dài.
            return self._chunk_section(None, text)

        sections: list[tuple[str | None, str]] = []

        # Phần trước heading đầu tiên (nếu có nội dung đáng kể).
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading_line = m.group().strip()
            body = text[start:end].strip()
            sections.append((heading_line, body))

        chunks: list[str] = []
        for heading, body in sections:
            chunks.extend(self._chunk_section(heading, body))
        return chunks

    def _chunk_section(self, heading: str | None, section_text: str) -> list[str]:
        section_text = section_text.strip()
        if not section_text:
            return []

        if len(section_text) <= self.chunk_size:
            return [section_text]

        # Section quá dài -> hạ xuống recursive, rồi gắn lại heading vào mỗi mảnh
        # con để mỗi chunk vẫn tự mang đủ ngữ cảnh (biết mình thuộc heading nào).
        pieces = self._fallback.chunk(section_text)
        if not heading:
            return pieces

        result = []
        for piece in pieces:
            if piece.strip().startswith(heading):
                result.append(piece)
            else:
                result.append(f"{heading}\n{piece}")
        return result


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