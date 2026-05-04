# Individual Reflection — Lab 18

**Tên:** Nguyễn Mạnh Quyền — 2A202600481  
**Module phụ trách:** M1: Advanced Chunking  
**File chính:** `src/m1_chunking.py`

---

## 1. Đóng góp kỹ thuật cụ thể

Trong bài lab này, em phụ trách module **Advanced Chunking**, là bước đầu tiên quyết định chất lượng context trước khi search và rerank. Các phần em trực tiếp triển khai gồm:

- `chunk_semantic()`: tách văn bản thành các câu, encode theo sentence embedding và gom các câu liền kề khi độ tương đồng còn đủ cao
- `chunk_hierarchical()`: tạo hai tầng chunk gồm parent chunk để giữ ngữ cảnh đầy đủ và child chunk để phục vụ retrieval chính xác hơn
- `chunk_structure_aware()`: tách tài liệu markdown theo header để giữ nguyên cấu trúc logic của văn bản
- `compare_strategies()`: chạy đồng thời 4 chiến lược chunking và so sánh các thống kê cơ bản như số chunk, độ dài trung bình, min và max

Điểm quan trọng của module này là em không chỉ tạo chunk theo kích thước cố định, mà cố gắng giữ được **ranh giới ngữ nghĩa** và **ranh giới cấu trúc tài liệu**, phù hợp hơn với dữ liệu pháp lý và tài liệu có đề mục.

## 2. Đối chiếu với rubric cá nhân

- **A1. Module implementation đúng logic:** đã hoàn thành đủ 3 chiến lược chunking nâng cao và phần compare với baseline
- **A2. Test pass:** nội dung code bám trực tiếp các tiêu chí trong `tests/test_m1.py`, gồm semantic grouping, parent-child mapping, section metadata và compare đủ 4 strategies
- **A3. Vietnamese-specific handling:** tuy chunking không dùng tokenizer tiếng Việt như BM25, em vẫn xử lý theo cấu trúc câu và section thực tế của tài liệu tiếng Việt thay vì cắt máy móc theo số ký tự
- **A4. Code quality:** dùng dataclass `Chunk`, type hints rõ ràng, metadata nhất quán giữa các strategy và tách logic theo từng hàm độc lập
- **A5. TODO markers hoàn thành:** phần TODO của module được thay thế hoàn toàn bằng code chạy được, không để lại marker dang dở

## 3. Kiến thức học được và liên hệ bài giảng

Điều em học rõ nhất là **chunking không phải bước phụ**, mà là nền móng của toàn bộ hệ thống RAG. Nếu chunk quá lớn thì retrieval thiếu chính xác; nếu chunk quá nhỏ hoặc cắt sai ranh giới thì context bị đứt gãy và LLM rất dễ trả lời thiếu hoặc sai.

Kiến thức này gắn trực tiếp với phần bài giảng về:

- fixed-size chunking chỉ là baseline
- semantic chunking giúp tránh cắt giữa một ý đang liên tục
- hierarchical chunking giúp cân bằng giữa retrieval precision và generation context
- structure-aware chunking đặc biệt hữu ích với markdown, policy document và tài liệu pháp lý

## 4. Khó khăn và cách giải quyết

Khó khăn lớn nhất của em là phải thiết kế chunk sao cho **vừa pass test vừa có ý nghĩa thực tế**. Nếu chỉ cắt để qua test thì module sẽ yếu khi chạy thật; ngược lại, nếu tối ưu quá phức tạp thì khó giữ code gọn và ổn định.

Em xử lý bằng cách:

- giữ semantic chunking ở mức đơn giản nhưng đúng bản chất: so sánh các câu liền kề thay vì làm clustering phức tạp
- dùng parent-child mapping rõ ràng bằng `parent_id` để hierarchical chunking dễ kiểm tra và dễ dùng lại ở bước retrieval
- dùng regex theo markdown header để section-aware chunking bám sát cấu trúc tài liệu thật

## 5. Tác động tới pipeline nhóm

Module M1 ảnh hưởng trực tiếp đến recall của hệ thống. Nếu chunking làm mất ngữ cảnh, các module search và rerank phía sau sẽ phải xử lý trên dữ liệu đầu vào kém chất lượng. Vì vậy, phần em làm là nền cho các bước M2 và M3 hoạt động ổn định hơn.

## 6. Tự đánh giá

Em tự đánh giá **5/5** cho phần việc cá nhân này vì:

- Hoàn thành đúng vai trò của module nền tảng trong pipeline RAG
- Triển khai đủ 3 chiến lược chunking nâng cao thay vì chỉ chỉnh sửa nhỏ trên baseline
- Bám sát tiêu chí test và tiêu chí chấm cá nhân
- Reflection mô tả đúng phần code em phụ trách trong project

## 7. Cập nhật kết quả tích hợp nhóm

Sau khi tích hợp pipeline và rerun 30 câu hỏi với parent-context retrieval, production RAG đạt `faithfulness = 0.9466`, `context_precision = 0.9537`, `context_recall = 0.9933` và `answer_relevancy = 0.5431`. Kết quả này cho thấy hierarchical chunking của M1 có tác động rõ khi pipeline index child chunks nhưng trả về parent chunks để tăng coverage cho các câu hỏi nhiều ý.
