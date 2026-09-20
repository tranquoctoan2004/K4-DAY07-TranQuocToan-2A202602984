from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        # Store rỗng: báo rõ ràng, không gọi LLM vô ích.
        if self.store.get_collection_size() == 0:
            return "Cơ sở tri thức đang trống nên chưa thể trả lời. Hãy nạp tài liệu trước."

        # 1) Truy xuất top-k (có thể lọc theo metadata, ví dụ {"audience": "buyer"}).
        if metadata_filter:
            results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.store.search(question, top_k=top_k)

        if not results:
            return "Không tìm thấy tài liệu nào phù hợp với câu hỏi (và bộ lọc) đã cho."

        # 2) Dựng ngữ cảnh: mỗi chunk đánh số [1], [2], [3] kèm nguồn để truy vết.
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("doc_id", result.get("id", "unknown"))
            url = metadata.get("source_url")
            source_text = f"{source} | {url}" if url else str(source)
            context_blocks.append(f"[{index}] (nguồn: {source_text})\n{result['content']}")
        context = "\n\n".join(context_blocks)

        # 3) Prompt: chỉ dùng ngữ cảnh, trả lời "không tìm thấy" nếu thiếu, bắt buộc trích dẫn.
        prompt = (
            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu.\n"
            "Chỉ sử dụng thông tin trong phần NGỮ CẢNH bên dưới. Không dùng kiến thức bên ngoài.\n"
            "Nếu ngữ cảnh không đủ để trả lời, hãy trả lời: \"Không tìm thấy thông tin trong tài liệu.\"\n"
            "Khi trả lời, trích dẫn số của đoạn đã dùng, ví dụ [1] hoặc [2].\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI:"
        )
        return self.llm_fn(prompt)