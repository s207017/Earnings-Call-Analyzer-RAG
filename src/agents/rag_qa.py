"""RAG-based Q&A engine for earnings call transcripts."""

import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in earnings call analysis.
You answer questions based ONLY on the provided earnings call transcript excerpts.

Rules:
1. Only use information from the provided context. Do not use prior knowledge about these companies.
2. Cite your sources using [1], [2], etc. matching the numbered context excerpts.
3. If the context doesn't contain enough information, say so explicitly.
4. Be specific with numbers, dates, and quotes when available.
5. Distinguish between management statements (prepared remarks) and analyst Q&A responses.
6. Note the speaker and their role when relevant.
"""

# Financial query expansion patterns — maps vague terms to specific sub-queries
QUERY_EXPANSIONS = {
    r"\bhow did (they|the company|it) do\b": [
        "{query}", "revenue growth earnings", "guidance outlook",
    ],
    r"\bperformance\b": [
        "{query}", "revenue earnings growth margins",
    ],
    r"\boutlook|guidance|forecast\b": [
        "{query}", "expect anticipate next quarter year guidance",
    ],
    r"\brisk|concern|worry\b": [
        "{query}", "challenge headwind risk pressure decline",
    ],
    r"\bstrategy|plan|initiative\b": [
        "{query}", "invest growth strategy plan initiative opportunity",
    ],
    r"\bcompetition|competitive\b": [
        "{query}", "market share competitive advantage differentiation",
    ],
}


class RAGEngine:
    """RAG Q&A engine supporting Ollama (local), OpenAI, and Anthropic LLMs."""

    def __init__(self, retrieval_engine, llm_provider: str = None,
                 project_root: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.retrieval = retrieval_engine
        self.llm_provider = llm_provider or self.config["models"].get("llm_provider", "ollama")
        self.llm_model = self.config["models"].get("llm_model", "llama3.2:3b")
        self.conversation_history = []
        self._client = None

    def _get_client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            if self.llm_provider == "ollama":
                import requests
                # Verify Ollama is running
                try:
                    requests.get("http://localhost:11434/api/tags", timeout=2)
                except requests.ConnectionError:
                    raise ConnectionError("Ollama is not running. Start it with: ollama serve")
                self._client = "ollama"  # Use requests directly
            elif self.llm_provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            elif self.llm_provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        return self._client

    def _format_context(self, sources: List[Dict]) -> str:
        """Format retrieved chunks as numbered context for the LLM."""
        parts = []
        for i, src in enumerate(sources, 1):
            header = f"[{i}] {src['ticker']} {src['quarter']} | {src.get('speaker', 'Unknown')} ({src.get('role', 'Unknown')}) | {src.get('section', '')}"
            parts.append(f"{header}\n{src['text']}")
        return "\n\n---\n\n".join(parts)

    def _truncate_context(self, sources: List[Dict], max_tokens: int = None) -> List[Dict]:
        """Truncate sources to fit within context window."""
        if max_tokens is None:
            max_tokens = self.config["retrieval"]["max_context_tokens"]
        truncated = []
        total = 0
        for src in sources:
            token_est = len(src["text"].split())
            if total + token_est > max_tokens:
                break
            truncated.append(src)
            total += token_est
        return truncated

    def _call_ollama(self, system: str, user_message: str) -> str:
        """Call local Ollama model."""
        import requests

        messages = [{"role": "system", "content": system}]
        for h in self.conversation_history[-4:]:
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.llm_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1500},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _call_openai(self, system: str, user_message: str) -> str:
        """Call OpenAI API."""
        client = self._get_client()
        messages = [{"role": "system", "content": system}]
        # Add conversation history
        for h in self.conversation_history[-4:]:  # Keep last 4 turns
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system: str, user_message: str) -> str:
        """Call Anthropic API."""
        client = self._get_client()
        messages = []
        for h in self.conversation_history[-4:]:
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            system=system,
            messages=messages,
            max_tokens=1500,
            temperature=0.1,
        )
        return response.content[0].text

    def _extract_citations(self, answer: str) -> List[int]:
        """Extract citation indices from answer text."""
        return sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer)))

    def _expand_query(self, query: str) -> List[str]:
        """Expand vague queries into multiple specific sub-queries for better retrieval."""
        queries = [query]
        query_lower = query.lower()

        for pattern, expansions in QUERY_EXPANSIONS.items():
            if re.search(pattern, query_lower):
                for expansion in expansions:
                    expanded = expansion.format(query=query)
                    if expanded != query:
                        queries.append(expanded)
                break  # only apply first matching pattern

        return queries

    def _multi_query_retrieve(self, queries: List[str], top_k: int,
                              filters: Dict = None) -> List[Dict]:
        """Retrieve from multiple queries and deduplicate by chunk_id."""
        seen = set()
        all_results = []

        for q in queries:
            results = self.retrieval.hybrid_search(
                q, top_k=top_k, filters=filters, rerank=True)
            for r in results:
                cid = r["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    all_results.append(r)

        # Re-rank combined results against original query
        if len(queries) > 1 and self.retrieval.rerank_enabled:
            all_results = self.retrieval.rerank(queries[0], all_results, top_k)

        return all_results[:top_k]

    def answer(self, question: str, filters: Dict = None,
               top_k: int = 5) -> Dict:
        """Retrieve relevant chunks and generate answer with citations.

        Uses query expansion for vague questions, cross-encoder re-ranking,
        and metadata filtering.
        """
        # Expand query for better coverage
        queries = self._expand_query(question)

        # Retrieve with multi-query + re-ranking
        sources = self._multi_query_retrieve(queries, top_k=top_k * 2, filters=filters)
        sources = self._truncate_context(sources)

        if not sources:
            return {
                "answer": "No relevant transcript excerpts found for this query.",
                "sources": [],
                "model_used": self.llm_provider,
            }

        context = self._format_context(sources)
        user_message = f"Context from earnings call transcripts:\n\n{context}\n\nQuestion: {question}"

        # Generate
        try:
            if self.llm_provider == "ollama":
                answer_text = self._call_ollama(SYSTEM_PROMPT, user_message)
            elif self.llm_provider == "openai":
                answer_text = self._call_openai(SYSTEM_PROMPT, user_message)
            elif self.llm_provider == "anthropic":
                answer_text = self._call_anthropic(SYSTEM_PROMPT, user_message)
            else:
                answer_text = "LLM provider not configured."
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer_text = f"Error generating answer: {e}"

        # Track conversation
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer_text})

        # Map citations to sources
        cited_indices = self._extract_citations(answer_text)
        cited_sources = []
        for i in cited_indices:
            if 1 <= i <= len(sources):
                src = sources[i - 1].copy()
                cited_sources.append(src)

        return {
            "answer": answer_text,
            "sources": sources[:top_k],
            "cited_sources": cited_sources,
            "model_used": f"{self.llm_provider}/{self.llm_model}",
            "queries_used": queries,
        }

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    from src.agents.retrieval import RetrievalEngine

    engine = RetrievalEngine()
    try:
        engine.load_index()
        rag = RAGEngine(engine)
        result = rag.answer("What was Apple's revenue in the latest quarter?")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])}")
    except Exception as e:
        print(f"Error: {e}. Build index first.")
