"""Retrieval engine with hybrid dense+sparse search for earnings call RAG."""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid retrieval engine combining FAISS dense search with BM25 sparse search."""

    def __init__(self, project_root: str = None,
                 embedding_model: str = None):
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = Path(project_root)

        config_path = self.project_root / "configs" / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        if embedding_model is None:
            embedding_model = self.config["models"]["embedding_model"]

        self.embedding_model_name = embedding_model
        self._encoder = None
        self.index = None
        self.bm25 = None
        self.chunks_df = None
        self.embeddings = None
        self.dense_weight = self.config["retrieval"]["dense_weight"]
        self.sparse_weight = self.config["retrieval"]["sparse_weight"]

    def _load_encoder(self):
        """Lazy-load sentence transformer model."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def build_index(self, chunks_df: pd.DataFrame, batch_size: int = 64):
        """Build FAISS index and BM25 index from chunks."""
        import faiss
        from rank_bm25 import BM25Okapi

        self.chunks_df = chunks_df.reset_index(drop=True)
        texts = self.chunks_df["text"].tolist()

        # Dense embeddings
        encoder = self._load_encoder()
        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            batch_emb = encoder.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings.append(batch_emb)

        self.embeddings = np.vstack(embeddings).astype("float32")
        dim = self.embeddings.shape[1]

        # FAISS index
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        # BM25 index
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """FAISS similarity search."""
        encoder = self._load_encoder()
        query_emb = encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.chunks_df.iloc[idx]
            results.append({
                "chunk_id": row["chunk_id"],
                "ticker": row["ticker"],
                "quarter": row["quarter"],
                "speaker": row["speaker"],
                "role": row["role"],
                "section": row["section"],
                "text": row["text"],
                "relevance_score": float(score),
                "source": "dense",
            })
        return results

    def sparse_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 sparse search."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            row = self.chunks_df.iloc[idx]
            results.append({
                "chunk_id": row["chunk_id"],
                "ticker": row["ticker"],
                "quarter": row["quarter"],
                "speaker": row["speaker"],
                "role": row["role"],
                "section": row["section"],
                "text": row["text"],
                "relevance_score": float(scores[idx]),
                "source": "sparse",
            })
        return results

    def hybrid_search(self, query: str, top_k: int = 10,
                      filters: Dict = None) -> List[Dict]:
        """Combine dense + sparse search using Reciprocal Rank Fusion."""
        dense_k = top_k * 3
        sparse_k = top_k * 3

        dense_results = self.dense_search(query, dense_k)
        sparse_results = self.sparse_search(query, sparse_k)

        # Reciprocal Rank Fusion
        rrf_k = 60  # standard RRF constant
        scores = {}
        chunk_data = {}

        for rank, r in enumerate(dense_results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + self.dense_weight / (rrf_k + rank + 1)
            chunk_data[cid] = r

        for rank, r in enumerate(sparse_results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + self.sparse_weight / (rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = r

        # Sort by RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in ranked:
            r = chunk_data[cid].copy()
            r["relevance_score"] = score
            r["source"] = "hybrid"
            results.append(r)

        # Apply metadata filters
        if filters:
            results = self.filter_by_metadata(results, filters)

        return results[:top_k]

    def filter_by_metadata(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Filter results by metadata fields (ticker, quarter, role, section)."""
        filtered = []
        for r in results:
            match = True
            for key, value in filters.items():
                if key in r:
                    if isinstance(value, list):
                        if r[key] not in value:
                            match = False
                    elif r[key] != value:
                        match = False
            if match:
                filtered.append(r)
        return filtered

    def save_index(self, path: str = None):
        """Save FAISS index and metadata to disk."""
        import faiss

        if path is None:
            path = str(self.project_root / self.config["paths"]["embeddings"])
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        self.chunks_df.to_parquet(save_dir / "chunks_meta.parquet", index=False)
        np.save(save_dir / "embeddings.npy", self.embeddings)
        with open(save_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"Index saved to {save_dir}")

    def load_index(self, path: str = None):
        """Load FAISS index and metadata from disk."""
        import faiss

        if path is None:
            path = str(self.project_root / self.config["paths"]["embeddings"])
        load_dir = Path(path)

        self.index = faiss.read_index(str(load_dir / "faiss.index"))
        self.chunks_df = pd.read_parquet(load_dir / "chunks_meta.parquet")
        self.embeddings = np.load(load_dir / "embeddings.npy")
        with open(load_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info(f"Index loaded from {load_dir}: {self.index.ntotal} vectors")

    def evaluate(self, queries: List[str], ground_truth: List[List[str]],
                 k_values: List[int] = None) -> Dict:
        """Evaluate retrieval with Recall@K and MRR."""
        if k_values is None:
            k_values = [3, 5, 10]

        metrics = {f"recall@{k}": [] for k in k_values}
        metrics["mrr"] = []

        for query, relevant_ids in zip(queries, ground_truth):
            results = self.hybrid_search(query, top_k=max(k_values))
            retrieved_ids = [r["chunk_id"] for r in results]

            # Recall@K
            for k in k_values:
                top_k_ids = set(retrieved_ids[:k])
                recall = len(top_k_ids & set(relevant_ids)) / max(len(relevant_ids), 1)
                metrics[f"recall@{k}"].append(recall)

            # MRR
            rr = 0.0
            for i, rid in enumerate(retrieved_ids):
                if rid in relevant_ids:
                    rr = 1.0 / (i + 1)
                    break
            metrics["mrr"].append(rr)

        return {k: np.mean(v) for k, v in metrics.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    engine = RetrievalEngine()
    chunks_path = Path(engine.project_root) / "data" / "processed" / "all_chunks.parquet"
    if chunks_path.exists():
        df = pd.read_parquet(chunks_path)
        engine.build_index(df)
        engine.save_index()

        results = engine.hybrid_search("What was Apple's revenue guidance?", top_k=5)
        for r in results:
            print(f"[{r['relevance_score']:.4f}] {r['ticker']} {r['quarter']} - {r['text'][:100]}...")
    else:
        print("No chunks found. Run data pipeline first.")
