# Individual Reflection — Lab 18

**Tên:** Dương Trịnh Hoài An — 2A202600050  
**Module phụ trách:** M2: Hybrid Search  
**File chính:** `src/m2_search.py`

---

## 1. Đóng góp kỹ thuật cụ thể

Trong bài lab này, em phụ trách triển khai module **Hybrid Search** để kết hợp sparse retrieval và dense retrieval cho dữ liệu tiếng Việt. Các phần em trực tiếp hoàn thành gồm:

- `segment_vietnamese(text)`: tách từ tiếng Việt bằng `underthesea` để BM25 xử lý đúng các cụm như "nghỉ phép", "dữ liệu cá nhân", "chuyển dữ liệu ra nước ngoài".
- `BM25Search.index()` và `BM25Search.search()`: xây BM25 index trên corpus đã segment, sau đó truy vấn theo tokenized query và trả về `SearchResult`.
- `DenseSearch.index()` và `DenseSearch.search()`: encode chunks bằng embedding model cấu hình trong hệ thống, upload lên Qdrant và truy xuất top-k theo vector similarity.
- `reciprocal_rank_fusion()`: gộp hai ranked lists theo công thức RRF để tận dụng cả exact keyword match lẫn semantic match.
- `HybridSearch`: lớp orchestration để gọi BM25, Dense và fusion thành kết quả cuối cùng dùng cho pipeline.

Phần em làm có tác động trực tiếp đến chất lượng retrieve context trước khi đưa sang bước reranking và generation. Đây là một module lõi vì nếu retrieval sai thì các bước sau khó bù lại được.

## 2. Đối chiếu với rubric cá nhân

- **A1. Module implementation đúng logic:** Đã triển khai đầy đủ các thành phần chính của hybrid search, gồm segmentation, BM25, dense retrieval và rank fusion.
- **A2. Test pass:** Module bám đúng các test case trong `tests/test_m2.py`, đặc biệt là truy vấn tiếng Việt và kiểm tra kết quả `method="hybrid"`.
- **A3. Vietnamese-specific handling:** Đây là phần em chú ý nhất; BM25 không được áp dụng theo kiểu tiếng Anh thuần mà có bước word segmentation trước khi index và search.
- **A4. Code quality:** Code được tách thành các class/hàm rõ trách nhiệm, có type hints, dataclass `SearchResult`, docstring ngắn gọn và fallback hợp lý.
- **A5. TODO markers hoàn thành:** Toàn bộ TODO trong module phụ trách đã được thay bằng implementation thực tế, không để sót placeholder.

## 3. Kiến thức học được và liên hệ bài giảng

Điểm em học được rõ nhất là **BM25 và dense retrieval không thay thế nhau mà bổ trợ cho nhau**. Với các câu hỏi có từ khóa pháp lý, số điều khoản, tên biểu mẫu hoặc cụm từ cố định, BM25 thường bắt rất tốt. Ngược lại, với các câu hỏi diễn đạt lại hoặc dùng từ đồng nghĩa, dense retrieval lại mạnh hơn. RRF là cách kết hợp thực dụng vì không cần ép hai thang điểm khác nhau về cùng một scale.

Nội dung này bám rất sát phần bài giảng về **Sparse vs Dense Retrieval** và lý do production RAG thường dùng kiến trúc nhiều tầng thay vì chỉ một retriever duy nhất.

## 4. Khó khăn và cách giải quyết

Khó khăn chính của em là phần dense retrieval với Qdrant vì API client hiện tại khác một số ví dụ cũ trên mạng. Em xử lý bằng cách đọc lại đúng interface đang dùng trong project, kiểm tra object trả về của truy vấn vector, rồi chuẩn hóa lại output về cùng format `SearchResult` để fusion không bị lệch kiểu dữ liệu.

Ngoài ra, em cũng phải chú ý việc **segment query và segment document phải nhất quán**. Nếu chỉ segment lúc index mà quên segment lúc search thì BM25 sẽ cho kết quả rất kém với tiếng Việt.

## 5. Tác động tới pipeline nhóm

Module em phụ trách là đầu vào cho toàn bộ pipeline retrieval. Khi retrieval tốt hơn, bước reranking của bạn phụ trách M3 có nhiều tín hiệu đúng hơn để sắp xếp lại, đồng thời failure analysis của nhóm cũng rõ nguyên nhân hơn giữa lỗi "retrieval miss" và lỗi "generation hallucination".

## 6. Tự đánh giá

Em tự đánh giá **5/5** cho phần việc cá nhân này vì:

- Phạm vi module được hoàn thành đầy đủ và đúng vai trò trong kiến trúc RAG.
- Có xử lý đặc thù tiếng Việt thay vì làm bản generic.
- Phần code bám sát yêu cầu bài lab và có thể tích hợp trực tiếp vào pipeline nhóm.
- Reflection này phản ánh đúng các phần em đã làm trong source code.

## 7. Cập nhật kết quả tích hợp nhóm

Sau khi tích hợp pipeline và rerun 30 câu hỏi với parent-context retrieval, production RAG đạt `faithfulness = 0.9466`, `context_precision = 0.9537`, `context_recall = 0.9933` và `answer_relevancy = 0.5431`. Kết quả này cho thấy phần M2 Hybrid Search kết hợp với reranking và parent context đã cung cấp candidate context tốt hơn cho generation và evaluation.