"""
Module 5: Enrichment Pipeline
==============================
Enrich chunks BEFORE embedding: Summarize, HyQA, Contextual Prepend, Auto Metadata.
One-time indexing cost — improves ALL subsequent queries.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk with enrichment applied."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _llm(system: str, user: str, max_tokens: int = 150) -> str | None:
    """Call gpt-4o-mini; return None if unavailable."""
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    LLM: summarize in 2-3 sentences to reduce noise when embedding.
    Extractive fallback: first 2 sentences.
    """
    result = _llm(
        system="Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        user=text,
        max_tokens=150,
    )
    if result:
        return result
    # Extractive fallback
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate N questions this chunk can answer — bridges vocabulary gap at query time.
    Extractive fallback: template questions from bold terms.
    """
    result = _llm(
        system=(
            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
            "Trả về mỗi câu hỏi trên 1 dòng."
        ),
        user=text,
        max_tokens=200,
    )
    if result:
        questions = result.split("\n")
        return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]

    # Extractive fallback — find bold terms and Điều N references
    key_terms = re.findall(r"\*\*(.+?)\*\*|Điều \d+\.\s+\w+", text)
    return [f"{t.strip()} là gì?" for t in key_terms[:n_questions]] if key_terms else []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend 1-sentence context describing where this chunk sits in the document.
    Anthropic benchmark: reduces retrieval failure by 49%.
    """
    result = _llm(
        system=(
            "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và "
            "nói về chủ đề gì. Chỉ trả về 1 câu."
        ),
        user=f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if result:
        return f"{result}\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extracts topic, entities, category — enables rich filtering at search time.
    """
    result = _llm(
        system=(
            'Trích xuất metadata từ đoạn văn. Trả về JSON: '
            '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|legal", "language": "vi|en"}'
        ),
        user=text,
        max_tokens=150,
    )
    if result:
        try:
            raw = result.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            return json.loads(raw)
        except Exception:
            pass
    return {"topic": "", "entities": [], "category": "general", "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Apply enrichment techniques to every chunk before indexing.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Subset of ["summary", "hyqa", "contextual", "metadata", "full"]
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        text = chunk["text"]
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")

        use_all = "full" in methods
        summary = summarize_chunk(text) if use_all or "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if use_all or "hyqa" in methods else []
        enriched_text = contextual_prepend(text, source) if use_all or "contextual" in methods else text
        auto_meta = extract_metadata(text) if use_all or "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**meta, **auto_meta},
                method="+".join(methods),
            )
        )

    return enriched


if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary:  {summarize_chunk(sample)}\n")
    print(f"HyQA:     {generate_hypothesis_questions(sample)}\n")
    print(f"Context:  {contextual_prepend(sample, 'Sổ tay nhân viên VinUni')}\n")
    print(f"Metadata: {extract_metadata(sample)}")
