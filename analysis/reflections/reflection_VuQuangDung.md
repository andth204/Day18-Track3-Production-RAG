# Individual Reflection — Lab 18

**Tên:** Vũ Quang Dũng — 2A202600442  
**Module phụ trách:** M4: RAGAS Evaluation  
**File chính:** `src/m4_eval.py`

---

## 1. Đóng góp kỹ thuật cụ thể

Trong bài lab này, em phụ trách module **RAGAS Evaluation**, là phần giúp nhóm đo chất lượng hệ thống theo tiêu chí có cấu trúc thay vì chỉ nhìn cảm tính vào vài câu trả lời. Các phần em trực tiếp hoàn thành gồm:

- `load_test_set()`: đọc bộ câu hỏi đánh giá từ `test_set.json` và chuẩn hóa đầu vào cho pipeline
- `evaluate_ragas()`: dựng dữ liệu đánh giá từ `questions`, `answers`, `contexts`, `ground_truths`, gọi RAGAS evaluation và trả về report theo schema thống nhất của project
- `failure_analysis()`: lấy các trường hợp điểm thấp nhất, xác định metric tệ nhất và map sang diagnosis cùng suggested fix theo Diagnostic Tree
- `save_report()`: ghi kết quả aggregate và danh sách failure sang JSON để dùng cho báo cáo nhóm
- `EvalResult`: chuẩn hóa cấu trúc dữ liệu per-question để việc phân tích lỗi rõ ràng và nhất quán

Phần em làm tạo ra cầu nối giữa code pipeline và phần phân tích chất lượng. Nhờ đó nhóm không chỉ biết hệ thống "chạy được", mà còn biết **đang yếu ở retrieval, prompt hay faithfulness**.

## 2. Đối chiếu với rubric cá nhân

- **A1. Module implementation đúng logic:** đã hoàn thành luồng evaluate, phân tích lỗi và xuất report; đây là phần cốt lõi của M4
- **A2. Test pass:** module bám đúng các tiêu chí trong `tests/test_m4.py`, gồm load test set, trả về đủ metric keys dạng numeric và failure analysis có `diagnosis` cùng `suggested_fix`
- **A3. Vietnamese-specific handling:** evaluation được dùng trên bộ câu hỏi và ground truth tiếng Việt của project; phần failure analysis cũng được thiết kế để đọc ra nguyên nhân kỹ thuật có ý nghĩa với bài toán tiếng Việt
- **A4. Code quality:** có dataclass `EvalResult`, type hints, schema JSON rõ ràng và tách riêng evaluate/failure/report để dễ kiểm tra
- **A5. TODO markers hoàn thành:** phần TODO trong module đã được thay bằng logic cụ thể thay vì để khung trống

## 3. Kiến thức học được và liên hệ bài giảng

Điều em học được rõ nhất là **đánh giá RAG phải tách thành nhiều metric**, vì một câu trả lời sai có thể do nhiều nguyên nhân khác nhau. Nếu context thiếu thì là vấn đề retrieval/chunking; nếu context đúng mà answer vẫn bịa thì là vấn đề prompt hoặc generation. Chính vì vậy, failure analysis theo Diagnostic Tree có giá trị hơn nhiều so với việc chỉ nhìn một chỉ số tổng.

Nội dung này bám sát phần bài giảng về:

- dùng evaluation để biến việc tối ưu RAG thành quy trình có số liệu
- đọc metric để truy ngược nguyên nhân gốc
- dùng failure cases để quyết định nên ưu tiên cải thiện chunking, search, reranking hay prompt

## 4. Khó khăn và cách giải quyết

Khó khăn lớn nhất của em là tính ổn định của lớp evaluation, vì phần này phải xử lý đồng thời dữ liệu đầu vào, metric output và định dạng report cho các bước phân tích sau đó. Nếu schema không nhất quán thì report nhóm và failure analysis rất dễ bị lệch.

Em giải quyết bằng cách:

- chuẩn hóa dữ liệu theo `EvalResult` để các field nhất quán giữa evaluate và failure analysis
- giữ output dưới dạng dictionary có các metric key rõ ràng để pipeline và report dùng lại được
- thiết kế `failure_analysis()` theo hướng deterministic: lấy điểm trung bình để chọn bottom cases, rồi map metric thấp nhất sang diagnosis/fix cố định

## 5. Tác động tới pipeline nhóm

Module M4 là nơi biến kết quả chạy pipeline thành bằng chứng định lượng cho báo cáo. Nhờ module này, nhóm có thể chỉ ra metric nào đang mạnh, metric nào còn yếu, và đề xuất cải tiến tiếp theo dựa trên dữ liệu chứ không chỉ dựa trên cảm nhận khi demo.

## 6. Tự đánh giá

Em tự đánh giá **5/5** cho phần việc cá nhân này vì:

- Hoàn thành đúng vai trò đo lường và phân tích lỗi của hệ thống RAG
- Module có giá trị thực tế cho cả phần chấm điểm nhóm lẫn phần quyết định hướng cải thiện
- Phần code được tổ chức rõ ràng, có schema và failure mapping nhất quán
- Reflection này mô tả đúng các phần em đã phụ trách trong `src/m4_eval.py`

## 7. Cập nhật kết quả tích hợp nhóm

Sau khi tích hợp pipeline và rerun 30 câu hỏi với parent-context retrieval, production RAG đạt `faithfulness = 0.9466`, `context_precision = 0.9537`, `context_recall = 0.9933` và `answer_relevancy = 0.5431`. M4 đã được đồng bộ với RAGAS 0.4.3, lưu thêm per-question evidence trong report để failure analysis có câu hỏi, answer, context và ground truth rõ ràng hơn.
