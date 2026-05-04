# Group Report - Lab 18: Production RAG

**Nhóm:** An-Quyền-Đạt-Dũng  
**Ngày:** 2026-05-04

## Thành viên và phân công

| Tên | MSSV | Module | Trạng thái | Tests pass |
|-----|------|--------|------------|------------|
| Dương Trịnh Hoài An | 2A202600050 | M2: Hybrid Search | Hoàn thành | 5/5 |
| Nguyễn Mạnh Quyền | 2A202600481 | M1: Chunking | Hoàn thành | 13/13 |
| Nguyễn Tiến Đạt | 2A202600217 | M3: Reranking | Hoàn thành | 5/5 |
| Vũ Quang Dũng | 2A202600442 | M4: Evaluation | Hoàn thành | 4/4 |

## Pipeline Architecture

```text
M1 Hierarchical Chunking
  -> M2 Hybrid Search: BM25 + bge-m3 dense retrieval + RRF
  -> M3 CrossEncoder Reranker: bge-reranker-v2-m3
  -> Parent-context retrieval: index child chunks, return parent chunks for generation/evaluation
  -> LLM Generate: gpt-4o-mini, temperature=0.0, max_tokens=512
  -> M4 RAGAS Evaluation: faithfulness, answer_relevancy, context_precision, context_recall
```

M5 contextual prepend is integrated and can be enabled with `ENABLE_ENRICHMENT=1`. The submitted rerun used `ENABLE_ENRICHMENT=0` to avoid one-time LLM enrichment calls and focus on parent-context retrieval.

## RAGAS Results

Production scores are from `reports/ragas_report.json`, rerun on the first 30 questions with `EVAL_MAX_QUESTIONS=30` and `CONTEXT_TOP_K=5`. Baseline scores are from `reports/naive_baseline_report.json` on the original baseline artifact.

| Metric | Naive Baseline | Production 30Q | Delta |
|--------|----------------|----------------|-------|
| Faithfulness | 0.9463 | **0.9466** | **+0.0003** |
| Answer Relevancy | 0.0000 | 0.5431 | +0.5431 |
| Context Precision | 0.7883 | **0.9537** | **+0.1654** |
| Context Recall | 0.7968 | **0.9933** | **+0.1966** |

**Kết quả chính:** Production pipeline đạt 3/4 metrics trên 0.75. `faithfulness = 0.9466` đạt ngưỡng bonus 0.85, `context_precision = 0.9537` cho thấy context được lọc rất tốt, và `context_recall = 0.9933` cho thấy parent-context retrieval giải quyết được lỗi thiếu coverage.

## Key Improvements

1. **Parent-context retrieval:** Pipeline index child chunks để search chính xác, nhưng đưa parent chunks vào generation/evaluation. Thay đổi này tăng recall mạnh mà vẫn giữ precision cao.
2. **RAGAS evaluator fixed:** RAGAS 0.4.3 cần OpenAI embedding wrapper đúng API; sau khi sửa, `answer_relevancy` không còn bị 0 do lỗi setup.
3. **Evaluation runtime control:** Pipeline hỗ trợ `EVAL_MAX_QUESTIONS=30`, giúp rerun nhanh hơn nhưng vẫn đủ dài để có bằng chứng RAGAS.
4. **Evidence saved:** `reports/pipeline_predictions.json` lưu question, answer, contexts và ground truth từng câu để failure analysis có bằng chứng trực tiếp.

## Latency Breakdown

Số liệu lấy từ `reports/latency_report.json`.

| Step | Time |
|------|------|
| Chunking, one-time | 0.00s |
| Enrichment, one-time | 0.00s in submitted rerun |
| Indexing, one-time | 84.14s |
| Reranker load, one-time | 8.50s |
| Search avg/query | 205.75ms |
| Rerank avg/query | 5936.30ms |
| LLM generate avg/query | 3265.58ms |
| Total avg/query | 9407.63ms |

**Bottleneck:** Reranking remains the largest per-query cost. The current system favors answer quality and context coverage over latency.

## Presentation Notes

1. RAGAS: Production reaches `faithfulness = 0.9466`, `context_precision = 0.9537`, and `context_recall = 0.9933`.
2. Biggest win: Parent-context retrieval improves coverage for multi-part and list-style questions.
3. Case study: Questions such as "Liệt kê 5 quyền cơ bản..." benefit because parent context contains surrounding legal clauses, not only one short child chunk.
4. Next step: Improve `answer_relevancy` with Vietnamese-specific answer relevancy prompting or a Vietnamese embedding/evaluator setup.