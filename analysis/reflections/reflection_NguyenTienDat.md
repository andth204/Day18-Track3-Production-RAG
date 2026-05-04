# Individual Reflection — Lab 18

**Tên:** Nguyễn Tiến Đạt — 2A202600217  
**Module phụ trách:** M3: Reranking  
**File chính:** `src/m3_rerank.py`

---

## 1. Đóng góp kỹ thuật cụ thể

Trong bài lab này, em phụ trách module **Reranking**, tức là bước refine kết quả retrieval trước khi đưa context vào LLM. Các phần em trực tiếp triển khai gồm:

- `CrossEncoderReranker._load_model()`: lazy-load cross-encoder để tránh khởi tạo nặng ngay từ đầu
- `CrossEncoderReranker.rerank()`: chấm điểm từng cặp `(query, document)`, sắp xếp lại theo `rerank_score` và trả về top-k
- `CrossEncoderReranker._keyword_score()`: fallback khi model không load được, giúp module vẫn hoạt động ổn định trong môi trường hạn chế
- `FlashrankReranker.rerank()`: thêm một lựa chọn nhẹ hơn để so sánh hoặc dùng trong môi trường cần tốc độ
- `benchmark_reranker()`: đo `avg_ms`, `min_ms`, `max_ms` để lượng hóa trade-off giữa chất lượng và độ trễ

Phần em làm giúp pipeline chuyển từ cách "lấy top-k trực tiếp từ retrieval" sang cách **two-stage retrieval**, trong đó bước hai tập trung tăng precision của context.

## 2. Đối chiếu với rubric cá nhân

- **A1. Module implementation đúng logic:** đã hoàn thành đầy đủ reranking flow, gồm load model, score documents, sort lại kết quả và benchmark độ trễ
- **A2. Test pass:** code được viết theo đúng các tiêu chí trong `tests/test_m3.py`, gồm kiểu dữ liệu trả về, thứ tự score giảm dần, document liên quan đứng trước và benchmark có đủ trường thống kê
- **A3. Vietnamese-specific handling:** chọn mô hình reranker đa ngôn ngữ có thể xử lý tốt câu hỏi và tài liệu tiếng Việt, thay vì dùng một baseline chỉ phù hợp tiếng Anh
- **A4. Code quality:** có dataclass `RerankResult`, type hints, tách riêng fallback logic và benchmark logic để code dễ đọc, dễ tái sử dụng
- **A5. TODO markers hoàn thành:** các TODO trong module phụ trách đã được thay bằng implementation hoàn chỉnh

## 3. Kiến thức học được và liên hệ bài giảng

Điểm em học rõ nhất là sự khác nhau giữa **bi-encoder** và **cross-encoder**. Bi-encoder phù hợp cho retrieve nhanh trên toàn corpus, còn cross-encoder chậm hơn nhưng chính xác hơn vì xét trực tiếp từng cặp query-document. Điều này đúng với kiến trúc được học trên lớp: bước đầu ưu tiên recall, bước sau ưu tiên precision.

Reranking giúp hạn chế trường hợp top chunks có vẻ liên quan theo từ khóa nhưng không thực sự trả lời đúng trọng tâm câu hỏi. Trong pipeline production RAG, đây là bước rất đáng giá vì nó cải thiện chất lượng context mà không cần thay đổi toàn bộ retriever.

## 4. Khó khăn và cách giải quyết

Khó khăn chính là cân bằng giữa **độ chính xác** và **latency**. Cross-encoder cho chất lượng tốt hơn nhưng nặng hơn đáng kể so với retrieval thông thường. Nếu không thiết kế gọn, reranking có thể trở thành bottleneck của pipeline.

Em giải quyết bằng cách:

- chỉ rerank trên tập ứng viên nhỏ sau retrieval, không rerank toàn bộ corpus
- tách benchmark thành hàm riêng để đo latency rõ ràng thay vì cảm tính
- thêm fallback keyword scoring để module không bị gãy hoàn toàn khi model không sẵn sàng
- bổ sung `FlashrankReranker` như một phương án nhẹ hơn khi cần so sánh hoặc tối ưu tốc độ

## 5. Tác động tới pipeline nhóm

Reranking là lớp bảo vệ cuối trước bước generation. Khi M2 trả về một tập ứng viên đủ recall, M3 giúp đưa các chunk liên quan nhất lên đầu, từ đó làm giảm khả năng LLM đọc nhầm chunk không liên quan và trả lời lan man hoặc sai trọng tâm.

## 6. Tự đánh giá

Em tự đánh giá **5/5** cho phần việc cá nhân này vì:

- Module hoàn thành đúng vai trò second-stage ranking trong kiến trúc RAG
- Có cả phần chất lượng lẫn phần benchmark, không chỉ dừng ở implement một hàm sort đơn giản
- Có phương án fallback để tăng tính ổn định của hệ thống
- Reflection phản ánh đúng những gì em đã trực tiếp triển khai trong source code

## 7. Cập nhật kết quả tích hợp nhóm

Sau khi tích hợp pipeline và rerun 30 câu hỏi với parent-context retrieval, production RAG đạt `faithfulness = 0.9466`, `context_precision = 0.9537`, `context_recall = 0.9933` và `answer_relevancy = 0.5431`. Kết quả này cho thấy M3 reranking giúp giữ context liên quan ở top đầu, đồng thời kết hợp tốt với parent context để tăng recall mà vẫn giữ precision cao.
