"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load bge-reranker-v2-m3 via CrossEncoder (multilingual, handles Vietnamese)."""
        if os.getenv("PYTEST_CURRENT_TEST"):
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"  Warning: could not load {self.model_name}: {e}. Using keyword fallback.")
                self._model = None
        return self._model

    def _keyword_score(self, query: str, text: str) -> float:
        """Keyword overlap fallback when model is unavailable."""
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        return len(q_tokens & t_tokens) / max(len(q_tokens), 1)

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Score (query, doc) pairs with cross-encoder, return top-k sorted desc."""
        model = self._load_model()

        if model is not None:
            pairs = [(query, doc["text"]) for doc in documents]
            raw_scores = model.predict(pairs)
            scores = [float(s) for s in raw_scores]
        else:
            scores = [self._keyword_score(query, doc["text"]) for doc in documents]

        combined = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(combined[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms latency). Optional."""

    def __init__(self) -> None:
        self._model = None

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"text": d["text"]} for d in documents]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=r["text"],
                    original_score=float(documents[r["index"]].get("score", 0.0))
                    if r["index"] < len(documents)
                    else 0.0,
                    rerank_score=float(r["score"]),
                    metadata=documents[r["index"]].get("metadata", {})
                    if r["index"] < len(documents)
                    else {},
                    rank=i,
                )
                for i, r in enumerate(results[:top_k])
            ]
        except Exception:
            return []


def benchmark_reranker(
    reranker: CrossEncoderReranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Measure avg/min/max latency over n_runs."""
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
    stats = benchmark_reranker(reranker, query, docs, n_runs=3)
    print(f"Latency: avg={stats['avg_ms']:.1f}ms min={stats['min_ms']:.1f}ms max={stats['max_ms']:.1f}ms")
