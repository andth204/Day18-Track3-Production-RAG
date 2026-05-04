"""Production RAG Pipeline — Group assignment: integrate M1+M2+M3+M4+M5."""
# ruff: noqa: E402

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _configure_console() -> None:
    """Keep Vietnamese progress logs printable on Windows codepages."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


_configure_console()

from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import contextual_prepend
from config import OPENAI_API_KEY, RERANK_TOP_K

# Latency tracking across pipeline steps
_timings: dict[str, float] = {}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _context_text(result) -> str:
    return result.metadata.get("parent_text") or result.text


def _dedupe_contexts(reranked: list, fallback_results: list, top_k: int) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()
    for result in reranked:
        text = _context_text(result)
        if text not in seen:
            contexts.append(text)
            seen.add(text)
        if len(contexts) >= top_k:
            return contexts
    for result in fallback_results:
        text = _context_text(result)
        if text not in seen:
            contexts.append(text)
            seen.add(text)
        if len(contexts) >= top_k:
            break
    return contexts


def build_pipeline() -> tuple[HybridSearch, CrossEncoderReranker]:
    """Build production RAG pipeline with latency breakdown."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents...")
    t0 = time.perf_counter()
    docs = load_documents()
    all_chunks: list[dict] = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_texts = {
            parent.metadata.get("parent_id"): parent.text
            for parent in parents
            if parent.metadata.get("parent_id")
        }
        for child in children:
            parent_text = parent_texts.get(child.parent_id, child.text)
            all_chunks.append(
                {
                    "text": child.text,
                    "metadata": {
                        **child.metadata,
                        "parent_id": child.parent_id,
                        "parent_text": parent_text,
                    },
                }
            )
    _timings["chunking_s"] = time.perf_counter() - t0
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents  ({_timings['chunking_s']:.2f}s)")

    # Step 2: Enrichment (M5) — BONUS: Contextual Prepend (Anthropic style)
    # Adds 1-sentence context header to each chunk before embedding → reduces retrieval failure
    # Cache to disk so LLM calls (~6 min) only happen once
    _ENRICHMENT_CACHE = "reports/enriched_chunks_cache.json"
    print("\n[2/4] Enriching chunks (M5: contextual prepend)...")
    t0 = time.perf_counter()
    if not _env_flag("ENABLE_ENRICHMENT"):
        enriched_chunks = all_chunks
        print("  Skipped; set ENABLE_ENRICHMENT=1 to run contextual prepend")
    elif os.path.exists(_ENRICHMENT_CACHE):
        with open(_ENRICHMENT_CACHE, encoding="utf-8") as _f:
            enriched_chunks: list[dict] = json.load(_f)
        print(f"  Loaded {len(enriched_chunks)} enriched chunks from cache")
    else:
        enriched_chunks = []
        for chunk in all_chunks:
            enriched_text = contextual_prepend(
                chunk["text"],
                document_title=chunk["metadata"].get("source", ""),
            )
            enriched_chunks.append({"text": enriched_text, "metadata": chunk["metadata"]})
        os.makedirs("reports", exist_ok=True)
        with open(_ENRICHMENT_CACHE, "w", encoding="utf-8") as _f:
            json.dump(enriched_chunks, _f, ensure_ascii=False, indent=2)
        print(f"  Enriched & cached {len(enriched_chunks)} chunks")
    _timings["enrichment_s"] = time.perf_counter() - t0
    print(f"  ({_timings['enrichment_s']:.2f}s)")
    all_chunks = enriched_chunks

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense bge-m3)...")
    t0 = time.perf_counter()
    search = HybridSearch()
    search.index(all_chunks)
    _timings["indexing_s"] = time.perf_counter() - t0
    print(f"  Indexed {len(all_chunks)} chunks  ({_timings['indexing_s']:.2f}s)")

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker (bge-reranker-v2-m3)...")
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    reranker._load_model()
    _timings["reranker_load_s"] = time.perf_counter() - t0
    print(f"  Reranker ready  ({_timings['reranker_load_s']:.2f}s)")

    return search, reranker


def _generate_answer(query: str, contexts: list[str]) -> str:
    """Call gpt-4o-mini with tight grounding prompt for high faithfulness."""
    if not OPENAI_API_KEY:
        return contexts[0] if contexts else "Không tìm thấy thông tin."
    try:
        from openai import OpenAI

        context_str = "\n\n".join(f"[Nguồn {i+1}] {c}" for i, c in enumerate(contexts))
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý trả lời câu hỏi dựa HOÀN TOÀN vào các đoạn context được cung cấp.\n"
                        "NGUYÊN TẮC BẮT BUỘC:\n"
                        "1. Chỉ dùng thông tin có trong context — KHÔNG ĐƯỢC thêm kiến thức bên ngoài.\n"
                        "2. Nếu context không đủ thông tin để trả lời, trả lời: 'Không tìm thấy trong tài liệu.'\n"
                        "3. Trả lời trực tiếp, đầy đủ và chính xác bằng tiếng Việt.\n"
                        "4. Nếu câu hỏi yêu cầu liệt kê, hãy liệt kê ĐỦ các mục từ context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Các đoạn tài liệu tham khảo:\n{context_str}\n\nCâu hỏi: {query}\n\nHãy trả lời dựa trên tài liệu trên:",
                },
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return contexts[0] if contexts else "Không tìm thấy thông tin."


def run_query(
    query: str, search: HybridSearch, reranker: CrossEncoderReranker
) -> tuple[str, list[str]]:
    """Hybrid search → rerank → LLM generate."""
    t0 = time.perf_counter()
    results = search.search(query)
    _timings["search_s"] = _timings.get("search_s", 0) + (time.perf_counter() - t0)

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = time.perf_counter()
    context_top_k = _env_int("CONTEXT_TOP_K", 5)
    reranked = reranker.rerank(query, docs, top_k=max(RERANK_TOP_K, context_top_k))
    _timings["rerank_s"] = _timings.get("rerank_s", 0) + (time.perf_counter() - t0)

    contexts = _dedupe_contexts(reranked, results, top_k=context_top_k)

    t0 = time.perf_counter()
    answer = _generate_answer(query, contexts)
    _timings["generate_s"] = _timings.get("generate_s", 0) + (time.perf_counter() - t0)

    return answer, contexts


def evaluate_pipeline(
    search: HybridSearch, reranker: CrossEncoderReranker
) -> dict:
    """Run full evaluation on test set, save report with latency breakdown."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    max_questions = _env_int("EVAL_MAX_QUESTIONS", len(test_set))
    if max_questions < len(test_set):
        test_set = test_set[:max_questions]
        print(f"  Using first {len(test_set)} questions because EVAL_MAX_QUESTIONS={max_questions}")
    questions: list[str] = []
    answers: list[str] = []
    all_contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:55]}...")

    os.makedirs("reports", exist_ok=True)
    with open("reports/pipeline_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "question": q,
                    "answer": a,
                    "contexts": c,
                    "ground_truth": gt,
                }
                for q, a, c, gt in zip(questions, answers, all_contexts, ground_truths)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("  Saved predictions to reports/pipeline_predictions.json")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        mark = "✓" if s >= 0.75 else "✗"
        print(f"  {mark} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, path="reports/ragas_report.json")

    _print_latency_breakdown(len(test_set))
    return results


def _print_latency_breakdown(n_queries: int) -> None:
    """Print per-step latency table (bonus requirement)."""
    search_avg = _timings.get("search_s", 0) / max(n_queries, 1) * 1000
    rerank_avg = _timings.get("rerank_s", 0) / max(n_queries, 1) * 1000
    generate_avg = _timings.get("generate_s", 0) / max(n_queries, 1) * 1000

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"  {'Step':<25} {'Total (s)':>10} {'Per query (ms)':>15}")
    print("-" * 55)
    print(f"  {'Chunking':<25} {_timings.get('chunking_s', 0):>10.2f} {'(one-time)':>15}")
    print(f"  {'Enrichment':<25} {_timings.get('enrichment_s', 0):>10.2f} {'(one-time)':>15}")
    print(f"  {'Indexing':<25} {_timings.get('indexing_s', 0):>10.2f} {'(one-time)':>15}")
    print(f"  {'Reranker load':<25} {_timings.get('reranker_load_s', 0):>10.2f} {'(one-time)':>15}")
    print(f"  {'Search (avg)':<25} {'':>10} {search_avg:>14.1f}ms")
    print(f"  {'Rerank (avg)':<25} {'':>10} {rerank_avg:>14.1f}ms")
    print(f"  {'LLM generate (avg)':<25} {'':>10} {generate_avg:>14.1f}ms")
    print(f"  {'Total per query':<25} {'':>10} {search_avg+rerank_avg+generate_avg:>14.1f}ms")

    # Save to reports/
    latency_report = {
        "chunking_s": _timings.get("chunking_s", 0),
        "enrichment_s": _timings.get("enrichment_s", 0),
        "indexing_s": _timings.get("indexing_s", 0),
        "reranker_load_s": _timings.get("reranker_load_s", 0),
        "search_avg_ms": round(search_avg, 2),
        "rerank_avg_ms": round(rerank_avg, 2),
        "generate_avg_ms": round(generate_avg, 2),
        "total_per_query_ms": round(search_avg + rerank_avg + generate_avg, 2),
        "n_queries": n_queries,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_report.json", "w", encoding="utf-8") as f:
        json.dump(latency_report, f, indent=2)
    print("\n  Saved latency report to reports/latency_report.json")


if __name__ == "__main__":
    start = time.time()
    search_obj, reranker_obj = build_pipeline()
    evaluate_pipeline(search_obj, reranker_obj)
    print(f"\nTotal: {time.time() - start:.1f}s")
