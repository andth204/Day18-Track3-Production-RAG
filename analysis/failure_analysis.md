# Failure Analysis - Lab 18: Production RAG

**Nhóm:** An-Quyền-Đạt-Dũng  
**Thành viên:** Dương Trịnh Hoài An (M2), Nguyễn Mạnh Quyền (M1), Nguyễn Tiến Đạt (M3), Vũ Quang Dũng (M4)

## RAGAS Scores

Production scores are from `reports/ragas_report.json`, rerun on 30 questions with parent-context retrieval.

| Metric | Naive Baseline | Production 30Q | Delta |
|--------|----------------|----------------|-------|
| Faithfulness | 0.9463 | **0.9466** | **+0.0003** |
| Answer Relevancy | 0.0000 | 0.5431 | +0.5431 |
| Context Precision | 0.7883 | **0.9537** | **+0.1654** |
| Context Recall | 0.7968 | **0.9933** | **+0.1966** |

Production pipeline improved retrieval quality strongly. The main remaining weak metric is `answer_relevancy`, while `faithfulness`, `context_precision`, and `context_recall` are all above 0.75.

## Bottom-5 Failures

### #1

- **Question:** Sự im lặng hoặc không phản hồi của chủ thể dữ liệu có được coi là sự đồng ý không?
- **Worst metric:** `answer_relevancy`, score = 0.0
- **Error Tree:** 
  - Output đúng? Likely grounded, because faithfulness and recall are high overall.
  - Context đúng? Yes, the consent rule is in the retrieved legal context.
  - Query rewrite OK? Yes.
- **Likely root cause:** RAGAS answer relevancy prompt/embedding is not Vietnamese-specific and can under-score short direct Vietnamese answers.
- **Suggested fix:** Use Vietnamese-aware answer relevancy evaluator or customize the relevancy prompt language.

### #2

- **Question:** Bên Xử lý dữ liệu cá nhân có trách nhiệm gì khi kết thúc hợp đồng với Bên Kiểm soát dữ liệu?
- **Worst metric:** `answer_relevancy`, score = 0.4294
- **Error Tree:** 
  - Output đúng? Partial.
  - Context đúng? Likely yes.
  - Query rewrite OK? Yes.
- **Likely root cause:** The answer may include contract-ending obligations but not mirror the wording expected by the evaluator, reducing semantic relevancy.
- **Suggested fix:** Prompt the generator to answer in the exact scope of the question first, then add details only if context requires.

### #3

- **Question:** Bên Kiểm soát dữ liệu cá nhân là tổ chức/cá nhân có vai trò gì?
- **Worst metric:** `context_precision`, score = 0.5889
- **Error Tree:** 
  - Output đúng? Likely mostly correct.
  - Context đúng? Mixed; parent-context retrieval increases recall but may include extra surrounding definitions.
  - Query rewrite OK? Yes.
- **Likely root cause:** Parent chunks improve coverage but can include adjacent definitions, lowering precision for simple definition questions.
- **Suggested fix:** Use adaptive context mode: child context for simple definition questions, parent context for multi-part/list questions.

### #4

- **Question:** Biện pháp bảo vệ dữ liệu cá nhân nhạy cảm có gì khác biệt so với dữ liệu cơ bản?
- **Worst metric:** `answer_relevancy`, score = 0.5229
- **Error Tree:** 
  - Output đúng? Likely grounded but may be broad.
  - Context đúng? Yes, recall is improved by parent context.
  - Query rewrite OK? Yes.
- **Likely root cause:** Comparison questions need concise side-by-side answers; broad legal wording can look less directly relevant.
- **Suggested fix:** Add prompt instruction to format comparison answers as "basic data vs sensitive data".

### #5

- **Question:** Trong những trường hợp nào việc xử lý dữ liệu cá nhân KHÔNG cần sự đồng ý của chủ thể dữ liệu?
- **Worst metric:** `answer_relevancy`, score = 0.4454
- **Error Tree:** 
  - Output đúng? Likely partial-to-good.
  - Context đúng? Yes, parent retrieval helps cover multiple exceptions.
  - Query rewrite OK? Yes.
- **Likely root cause:** Multi-item exception answers are long; answer relevancy can drop if the generated answer includes more clauses than the evaluator expects.
- **Suggested fix:** For list-many-items questions, produce numbered bullets with only the requested cases and no extra explanation.

## Failure Patterns

| Pattern | Diagnosis | Module cần ưu tiên |
|---------|-----------|--------------------|
| Answer relevancy thấp | Evaluator/prompt chưa tối ưu cho tiếng Việt và câu trả lời dài | M4 + generation prompt |
| Simple definition questions có lower precision | Parent context quá rộng với câu hỏi đơn giản | Pipeline adaptive context |
| Comparison questions | Need stricter answer format | Generation prompt |
| Multi-item legal questions | Need numbered direct answers | Generation prompt |

## Case Study

**Question:** Bên Kiểm soát dữ liệu cá nhân là tổ chức/cá nhân có vai trò gì?

**Error Tree walkthrough:**
1. Output đúng? Likely correct but should be checked against saved prediction.
2. Context đúng? Relevant context is retrieved, but parent chunks include adjacent legal definitions.
3. Query rewrite OK? Yes, this is a short definition question.
4. Root cause: Parent-context retrieval improves recall globally but is broader than necessary for a simple definition.
5. Fix: Use adaptive retrieval: if the question asks "là gì"/definition, pass child chunks; if it asks list/comparison/multi-step, pass parent chunks.

**Next step:** Add question classifier for context mode and customize answer relevancy evaluation for Vietnamese.