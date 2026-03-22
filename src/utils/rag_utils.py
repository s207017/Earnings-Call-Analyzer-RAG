"""RAG system utility functions."""

import re
from typing import List, Dict


def format_chunks_for_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks as numbered context for LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.get('ticker', '?')} {chunk.get('quarter', '?')} | {chunk.get('speaker', 'Unknown')} ({chunk.get('role', '')}) | {chunk.get('section', '')}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def extract_citations(answer_text: str) -> List[int]:
    """Parse [1], [2] citation references from LLM answer."""
    return sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text)))


def truncate_context(chunks: List[Dict], max_tokens: int = 3000) -> List[Dict]:
    """Truncate chunks list to fit within token budget."""
    truncated = []
    total = 0
    for chunk in chunks:
        tokens = len(chunk.get("text", "").split())
        if total + tokens > max_tokens:
            break
        truncated.append(chunk)
        total += tokens
    return truncated


def format_source_display(source: Dict) -> str:
    """Format a source chunk for UI display."""
    ticker = source.get("ticker", "?")
    quarter = source.get("quarter", "?")
    speaker = source.get("speaker", "Unknown")
    role = source.get("role", "")
    score = source.get("relevance_score", 0)
    text = source.get("text", "")[:200]
    return f"**{ticker} {quarter}** | {speaker} ({role}) | Score: {score:.3f}\n> {text}..."
