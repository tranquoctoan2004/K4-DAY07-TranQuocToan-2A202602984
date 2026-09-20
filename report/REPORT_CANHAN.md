# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Quốc Toản
**Nhóm:** [ĐIỀN TÊN NHÓM]
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.**

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

> Hai vector embedding có góc giữa chúng nhỏ (hướng gần giống nhau) bất kể độ dài vector, cho thấy hai đoạn văn bản mang ý nghĩa/ngữ cảnh tương đồng theo cách embedding biểu diễn.

**Ví dụ có độ tương tự CAO (về mặt ý nghĩa, nếu dùng embedding thật):**

- Câu A: "Đơn hàng của tôi được đổi trả trong bao lâu?"
- Câu B: "Thời gian đổi trả sản phẩm là bao nhiêu ngày?"
- Tại sao tương đồng: cùng hỏi về thời hạn đổi trả, chỉ khác cách diễn đạt (từ đồng nghĩa, đảo cấu trúc câu).

**Ví dụ có độ tương tự THẤP:**

- Câu A: "Thời tiết hôm nay ở Hà Nội thế nào?"
- Câu B: "Chính sách bảo hành pin xe máy điện ra sao?"
- Tại sao khác: hai chủ đề hoàn toàn không liên quan (thời tiết vs. chính sách bảo hành).

**Tại sao cosine phù hợp với text embedding hơn Euclidean distance?**

> Cosine chỉ quan tâm hướng của vector, không bị ảnh hưởng bởi độ dài văn bản (câu dài thường cho vector "to" hơn). Euclidean distance nhạy với độ lớn vector nên hai câu cùng nghĩa nhưng độ dài khác nhau có thể bị tính là "khác xa" dù hướng vector gần như nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunk?**

> Công thức: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> Kiểm tra lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk("x"*10000)` → **23 chunk**, khớp với công thức.

**Nếu overlap tăng lên 100, số chunk thay đổi thế nào?**

> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25` chunk (tăng từ 23 lên 25).
> Overlap lớn hơn làm bước trượt (`chunk_size - overlap`) nhỏ hơn nên cần nhiều chunk hơn để phủ hết văn bản. Overlap lớn hữu ích khi thông tin quan trọng có thể nằm vắt ngang ranh giới hai chunk (ví dụ một điều khoản bị cắt giữa chừng) — overlap giúp câu/ý đó xuất hiện trọn vẹn trong ít nhất một chunk, giảm mất ngữ cảnh khi truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chunking

**`SentenceChunker.chunk`**

> Dùng regex `(?<=[.!?])\s+|\n+` với lookbehind để tách sau dấu `.`, `!`, `?` mà vẫn giữ nguyên dấu câu trong chuỗi kết quả (lookbehind không "nuốt" ký tự đã match), đồng thời tách theo newline. Sau đó gom mỗi `max_sentences_per_chunk` câu liên tiếp thành một chunk bằng slicing theo bước `n`. Edge case: văn bản rỗng/toàn khoảng trắng trả về `[]`; các viết tắt như "TS." hay số thập phân "3.14" có thể bị tách nhầm vì regex chỉ dựa vào dấu câu, không phân biệt được ngữ cảnh viết tắt.

**`RecursiveChunker.chunk` / `_split`**

> Thuật toán đệ quy thử lần lượt các separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`: nếu đoạn văn đã đủ ngắn thì giữ nguyên; nếu separator hiện tại không xuất hiện trong đoạn thì thử separator nhỏ hơn; nếu tách được thì gắn lại separator vào cuối mỗi mảnh (trừ mảnh cuối) để không mất dấu câu/xuống dòng, rồi gom các mảnh liền kề lại tới gần `chunk_size` để tránh chunk vụn — mảnh nào sau khi gom vẫn còn quá dài thì đệ quy tiếp với separator nhỏ hơn. Base case là khi hết separator (`remaining_separators == []` hoặc gặp `""`) thì cắt cứng theo `chunk_size` ký tự (`_hard_cut`), đảm bảo đệ quy luôn dừng.

### Lớp EmbeddingStore

**`add_documents` + `search`**

> `add_documents` không tự chunk — mỗi `Document` truyền vào tương ứng đúng một record trong `self._store`, mỗi record chứa `id`, `content`, `metadata` (đã copy để tránh giữ chung tham chiếu với caller, và luôn có `metadata["doc_id"]`) và `embedding` tính từ `content`. `search` gọi hàm dùng chung `_search_records` để tính cosine similarity giữa embedding của query và embedding của toàn bộ record, sắp xếp giảm dần theo score rồi trả về top-k (không trả `embedding` ra ngoài để tránh lộ vector thô).

**`search_with_filter` + `delete_document`**

> `search_with_filter` lọc metadata **trước** (theo tất cả cặp key-value trong `metadata_filter`) để lấy tập ứng viên nhỏ hơn, sau đó mới chạy `_search_records` trên tập đã lọc — làm theo thứ tự này để tránh tính similarity lãng phí trên các record chắc chắn bị loại. `delete_document` xóa mọi record có `metadata["doc_id"]` khớp với `doc_id` truyền vào bằng list comprehension, so sánh kích thước store trước/sau để trả về `True`/`False`.

### Tác tử KnowledgeBaseAgent

**`answer`**

> Nếu store rỗng thì trả thông báo ngay, không gọi LLM. Ngược lại, truy xuất top-k (có hoặc không kèm `metadata_filter`), đánh số từng chunk `[1]`, `[2]`, `[3]` kèm nguồn (`doc_id` + `source_url` nếu có) để đảm bảo truy vết được nguồn. Prompt yêu cầu LLM chỉ dùng ngữ cảnh đã cho, trả lời "Không tìm thấy thông tin trong tài liệu." nếu ngữ cảnh không đủ, và bắt buộc trích dẫn số đoạn đã dùng (ví dụ `[1]`) để đảm bảo source traceability.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
(.venv) C:\Users\TQT\Documents\LabAI\K4-L3B-Data-Foundations>pytest tests/ -v
========================== test session starts ===========================
platform win32 -- Python 3.13.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\TQT\Documents\LabAI\K4-L3B-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\TQT\Documents\LabAI\K4-L3B-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

=========================== 42 passed in 0.63s ===========================
```

**Số lượng bài test vượt qua:** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

> Lưu ý: lab dùng `MockEmbedder` (băm MD5 nội dung), **không** hiểu ngữ nghĩa thật. Bảng dưới đây dự đoán theo trực giác ngữ nghĩa (nếu dùng embedding thật) rồi đối chiếu với điểm số mock thực tế để thấy rõ giới hạn của mock.

| Cặp | Câu A                                         | Câu B                                                 | Dự đoán | Điểm thực tế (mock) | Đúng?                                           |
| --- | --------------------------------------------- | ----------------------------------------------------- | ------- | ------------------- | ----------------------------------------------- |
| 1   | Đơn hàng của tôi được đổi trả trong bao lâu?  | Thời gian đổi trả sản phẩm là bao nhiêu ngày?         | cao     | -0.2182             | Sai                                             |
| 2   | Tiki bảo hành sản phẩm trong bao lâu?         | Thời gian bảo hành sản phẩm là bao lâu?               | cao     | -0.1554             | Sai                                             |
| 3   | Nhà Bán cần lưu video đóng gói bao lâu?       | Video đóng gói hàng hóa phải giữ trong thời gian nào? | cao     | 0.1314              | Đúng hướng (dương) nhưng số thấp                |
| 4   | Sản phẩm nào không được đổi trả theo nhu cầu? | Danh mục hàng hóa bị loại trừ đổi trả là gì?          | cao     | 0.0551              | Sai                                             |
| 5   | Thời tiết hôm nay ở Hà Nội thế nào?           | Chính sách bảo hành pin xe máy điện ra sao?           | thấp    | 0.0486              | Sai (điểm không thấp hơn hẳn các cặp liên quan) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về embeddings?**

> Bất ngờ nhất là cặp 5 (hai câu hoàn toàn không liên quan: thời tiết vs. bảo hành pin xe máy điện) có điểm dương (0.0486) gần bằng, thậm chí cao hơn cặp 4 (0.0551 — hai câu cùng chủ đề đổi trả). Trong khi đó cặp 1 và 2 — hai câu rõ ràng cùng hỏi một việc — lại có điểm **âm**. Điều này cho thấy `MockEmbedder` dựa trên băm MD5 của chuỗi ký tự, hoàn toàn không mã hoá ý nghĩa: hai câu diễn đạt khác nhau dù cùng nghĩa vẫn cho ra vector gần như ngẫu nhiên so với nhau. Để retrieval thật sự phản ánh ngữ nghĩa, cần dùng embedding thật (`LocalEmbedder`, `OpenAIEmbedder` hoặc `GeminiEmbedder`).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

> Chạy bằng `python bench.py` với chiến lược **HeadingChunker** (chunk theo heading `##`/`###`, tổng 116 chunk từ 5 tài liệu).

| #   | Câu hỏi                                                          | Top-1 Chunk truy xuất được (tóm tắt)                                  | Score | Có liên quan không?                                                        | Câu trả lời của Agent (tóm tắt)                        |
| --- | ---------------------------------------------------------------- | --------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------- | ------------------------------------------------------ |
| 1   | Thời gian hỗ trợ đổi trả hàng tại Tiki là bao nhiêu ngày?        | "Trường hợp Nhà Bán từ chối yêu cầu đổi–trả–bảo hành..." (seller-faq) | 0.298 | Không (đúng chủ đề chung, sai section — không có số ngày)                  | _(bench.py chỉ chạy retrieval, chưa gọi agent.answer)_ |
| 2   | Sản phẩm bảo hành gửi về Tiki mất bao lâu? (filter buyer)        | "Danh mục sản phẩm không áp dụng đổi trả theo nhu cầu..."             | 0.251 | Không (lạc đề dù đúng audience)                                            | —                                                      |
| 3   | Nhà Bán cần lưu video đóng gói bao lâu? (filter seller)          | "Trường hợp Nhà Bán từ chối yêu cầu..."                               | 0.214 | Không (câu trả lời thật nằm ở chunk khác trong cùng file, không lọt top-1) | —                                                      |
| 4   | Những sản phẩm nào không được đổi trả theo nhu cầu?              | "Bằng chứng và giải quyết tranh chấp..."                              | 0.256 | Không (đúng chủ đề đổi trả nhưng sai mục — mục đúng bị xếp ngoài top-3)    | —                                                      |
| 5   | Quy trình Nhà Bán xử lý khi nhận hàng trả là gì? (filter seller) | "Sau khi Nhà Bán 'Đồng ý' yêu cầu đổi trả, cần làm gì tiếp theo?"     | 0.376 | **Có** — đúng file, sát chủ đề quy trình xử lý                             | —                                                      |

**Bao nhiêu câu hỏi trả về chunk liên quan trong top-3?** 1 / 5 (chỉ query 5 có chunk đúng ở top-1; các câu còn lại đúng audience/chủ đề lớn nhưng sai section cụ thể — nguyên nhân chính là `MockEmbedder` không hiểu ngữ nghĩa, xem mục 4).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> So với baseline `ChunkingStrategyComparator` (fixed_size, by_sentences, recursive) chạy trên cùng corpus, `HeadingChunker` của tôi cho chunk bám sát cấu trúc thật của tài liệu hơn — mỗi chunk gần như trùng với một điều khoản hoặc một câu hỏi FAQ, trong khi `fixed_size` cắt cứng theo ký tự nên nhiều lần chia đôi giữa câu hoặc giữa hai mục khác nhau. Bài học lớn nhất là: **chọn chiến lược chunking phải dựa vào cấu trúc thật của dữ liệu**, không có một chiến lược "tốt nhất" chung cho mọi loại văn bản — với tài liệu chính sách/FAQ có heading rõ ràng như của nhóm, heading-based chunking là lựa chọn hợp lý hơn cắt cố định theo ký tự. Tuy nhiên điểm số retrieval thực tế trong `bench.py` (mục 5) cho thấy: chọn đúng chiến lược chunk chỉ giải quyết được vấn đề "chunk có mạch lạc hay không", còn kết quả tìm kiếm vẫn phụ thuộc rất nhiều vào chất lượng embedding — điều mà `MockEmbedder` không đáp ứng được.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------- |
| Khởi động (Warm-up)                             | 5 / 5            |
| Hướng tiếp cận của tôi (My Approach)            | 10 / 10          |
| Hoàn thiện code (Core Implementation — tests)   | 30 / 30          |
| Dự đoán độ tương tự (Similarity Predictions)    | 5 / 5            |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10           |
| **Tổng phần cá nhân**                           | **57 / 60**      |
