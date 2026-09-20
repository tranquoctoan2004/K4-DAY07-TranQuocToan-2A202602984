"""
bench.py — Giai đoạn 3: benchmark chiến lược retrieval trên corpus cá nhân.

Cách chạy:
    python bench.py

Mỗi thành viên chỉ nên đổi biến CHUNKER bên dưới để dùng chiến lược riêng
(fixed size / sentence / recursive / heading) rồi so sánh kết quả trong nhóm.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows mặc định dùng cp1252 khi stdout bị redirect ra file (> ket_qua...),
# cp1252 không mã hoá được tiếng Việt có dấu -> ép UTF-8 để tránh UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.chunking import (
    FixedSizeChunker,
    HeadingChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path("data/chinh-sach-doi-tra-tiki")

# ---------------------------------------------------------------------------
# 1) Chọn chiến lược chunk ở đây. Đổi đúng 1 dòng này để so sánh giữa các
#    thành viên trong nhóm (cùng corpus, cùng query, khác chunker).
# ---------------------------------------------------------------------------
CHUNKER = HeadingChunker(chunk_size=500)
# Các lựa chọn khác cho thành viên khác trong nhóm, ví dụ:
# CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# CHUNKER = RecursiveChunker(chunk_size=500)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tách YAML frontmatter đơn giản (key: value) và phần body Markdown."""
    if not text.startswith("---"):
        return {}, text

    _, fm_block, body = text.split("---", 2)
    metadata: dict = {}
    for line in fm_block.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"')
        metadata[key] = value
    return metadata, body.strip()


def load_documents(data_dir: Path, chunker) -> list[Document]:
    """Đọc mọi file .md, chunk body, tạo 1 Document / chunk với metadata trải đều."""
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw)
        chunks = chunker.chunk(body)
        for i, chunk_text in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}#{i}",
                    content=chunk_text,
                    metadata={**frontmatter, "doc_id": path.stem},
                )
            )
    return documents


# ---------------------------------------------------------------------------
# 2) Đúng 5 query benchmark (câu hỏi đa dạng: số liệu / điều kiện / quy trình /
#    liệt kê), có gold answer trích từ tài liệu thật. Ít nhất 1 query cần
#    metadata_filter theo audience để chứng minh filter có tác dụng.
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "question": "Thời gian hỗ trợ đổi trả hàng tại Tiki là bao nhiêu ngày?",
        "metadata_filter": None,
        "gold_answer": (
            "30 ngày kể từ lúc nhận hàng thành công (áp dụng từ đơn hàng đặt "
            "từ 15/04/2024); riêng Thiết bị số - Phụ kiện số, Điện gia dụng do "
            "Tiki Trading bán được đổi trả trong 365 ngày nếu lỗi kỹ thuật."
        ),
        "expected_doc_id": "buyer-chinh-sach-doi-tra-hoan-tien",
    },
    {
        "question": "Sản phẩm bảo hành gửi về Tiki thì mất bao lâu để nhận lại?",
        "metadata_filter": {"audience": "buyer"},
        "gold_answer": (
            "Khoảng 15-30 ngày tùy linh kiện thay thế (không tính thời gian "
            "vận chuyển); riêng sản phẩm Apple dự kiến 30-60 ngày."
        ),
        "expected_doc_id": "buyer-chinh-sach-bao-hanh",
    },
    {
        "question": "Nhà Bán cần lưu trữ video đóng gói hàng hóa trong bao lâu?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": (
            "Tối thiểu 45 ngày (30 ngày kể từ khi giao hàng thành công cộng "
            "thêm thời gian luân chuyển giao hàng và thu hồi sản phẩm)."
        ),
        "expected_doc_id": "seller-faq-doi-tra-bao-hanh",
    },
    {
        "question": "Những nhóm sản phẩm nào không được đổi trả theo nhu cầu?",
        "metadata_filter": None,
        "gold_answer": (
            "Ví dụ: đồ lót/đồ bơi, nước hoa, trang sức, báo/tạp chí và "
            "CD/DVD, hoa tươi/cây cảnh, voucher-dịch vụ, thực phẩm tươi "
            "sống/đông lạnh, hàng giao từ nước ngoài, quà tặng kèm."
        ),
        "expected_doc_id": "buyer-san-pham-han-che-doi-tra",
    },
    {
        "question": "Quy trình Nhà Bán xử lý khi nhận lại hàng trả từ khách hàng gồm những bước nào?",
        "metadata_filter": {"audience": "seller"},
        "gold_answer": (
            "Khách đóng gói trả về theo hướng dẫn; Nhà Bán khi nhận hàng từ "
            "đối tác vận chuyển phải đồng kiểm, quay clip mở sản phẩm; nếu "
            "hàng trả về lỗi/hư hỏng/sai sản phẩm thì giữ hàng, giữ clip và "
            "khiếu nại tại Seller Center để Tiki xử lý."
        ),
        "expected_doc_id": "seller-faq-doi-tra-bao-hanh",
    },
]


def run_benchmark() -> None:
    documents = load_documents(DATA_DIR, CHUNKER)
    store = EmbeddingStore(collection_name="bench_store")
    store.add_documents(documents)

    print(f"Chiến lược chunk: {CHUNKER.__class__.__name__}")
    print(f"Đã nạp {store.get_collection_size()} chunk từ {DATA_DIR}\n")

    for i, q in enumerate(QUERIES, start=1):
        print(f"=== Query {i}: {q['question']}")
        if q["metadata_filter"]:
            print(f"    metadata_filter={q['metadata_filter']}")
        print(f"    Gold answer: {q['gold_answer']}")

        results = store.search_with_filter(
            q["question"], top_k=3, metadata_filter=q["metadata_filter"]
        )
        if not results:
            print("    (Không có kết quả)")
        for rank, r in enumerate(results, start=1):
            print(
                f"    top{rank}: score={r['score']:.3f} "
                f"doc_id={r['metadata'].get('doc_id')} id={r['id']}"
            )
            preview = r["content"][:100].replace("\n", " ")
            print(f"           preview: {preview}...")
        print()


if __name__ == "__main__":
    run_benchmark()