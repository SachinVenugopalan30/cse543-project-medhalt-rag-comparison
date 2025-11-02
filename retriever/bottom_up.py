#!/usr/bin/env python3
"""
Bottom-up retrieval: Vector-based re-ranking of graph candidates.

Implements the second phase of U-Retrieval:
1. Take candidates from top-down phase
2. Compute semantic similarity scores
3. Combine graph scores with embedding scores
4. Re-rank and return top-k results
"""

import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BottomUpReranker:
    """Bottom-up vector-based re-ranking."""

    def __init__(
        self,
        index_manager,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        """
        Initialize bottom-up reranker.

        Args:
            index_manager: IndexManager instance (from index_utils)
            alpha: Weight for graph score
            beta: Weight for embedding score
        """
        self.index_manager = index_manager
        self.alpha = alpha
        self.beta = beta

        # Validate weights
        if not np.isclose(alpha + beta, 1.0):
            logger.warning(
                f"Alpha ({alpha}) + Beta ({beta}) != 1.0. "
                "Normalizing weights..."
            )
            total = alpha + beta
            self.alpha = alpha / total
            self.beta = beta / total

        logger.info(
            f"Initialized with weights: alpha={self.alpha:.2f}, beta={self.beta:.2f}"
        )

    def compute_embedding_scores(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute embedding similarity scores for candidates.

        Args:
            query: Query text
            candidates: List of candidate chunks from top-down phase

        Returns:
            Dictionary mapping chunk_id to embedding score
        """
        if not candidates:
            return {}

        # Extract candidate chunk IDs
        candidate_ids = [c['chunk_id'] for c in candidates]

        # Get full chunk data from index
        chunks = []
        for chunk_id in candidate_ids:
            chunk = self.index_manager.get_chunk_by_id(chunk_id)
            if chunk:
                chunks.append(chunk)

        if not chunks:
            logger.warning("No chunks found in index for candidates")
            return {}

        # Search index for these specific chunks
        # This is a constrained search within candidate set
        all_results = self.index_manager.search(
            query,
            top_k=len(chunks) * 2  # Get more than needed for safety
        )

        # Extract scores for our candidates
        embedding_scores = {}
        for result in all_results:
            chunk_id = result['chunk']['id']
            if chunk_id in candidate_ids:
                # Convert distance to similarity score
                # Assuming normalized embeddings, score is already similarity
                embedding_scores[chunk_id] = result['score']

        # Normalize scores to 0-1 range
        if embedding_scores:
            max_score = max(embedding_scores.values())
            min_score = min(embedding_scores.values())

            if max_score > min_score:
                for chunk_id in embedding_scores:
                    normalized = (embedding_scores[chunk_id] - min_score) / \
                                (max_score - min_score)
                    embedding_scores[chunk_id] = normalized
            else:
                # All scores are the same
                for chunk_id in embedding_scores:
                    embedding_scores[chunk_id] = 1.0

        logger.debug(
            f"Computed embedding scores for {len(embedding_scores)} candidates"
        )

        return embedding_scores

    def combine_scores(
        self,
        candidates: List[Dict[str, Any]],
        embedding_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Combine graph and embedding scores.

        Args:
            candidates: Candidates with graph_score
            embedding_scores: Chunk ID -> embedding score mapping

        Returns:
            Candidates with combined scores
        """
        for candidate in candidates:
            chunk_id = candidate['chunk_id']
            graph_score = candidate.get('graph_score', 0.0)
            embedding_score = embedding_scores.get(chunk_id, 0.0)

            # Combined score: weighted sum
            combined_score = (
                self.alpha * graph_score +
                self.beta * embedding_score
            )

            candidate['embedding_score'] = embedding_score
            candidate['combined_score'] = combined_score

        return candidates

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using combined graph + embedding scores.

        Args:
            query: Query text
            candidates: Candidates from top-down phase
            top_k: Number of results to return

        Returns:
            Re-ranked top-k candidates
        """
        if not candidates:
            return []

        logger.info(f"Re-ranking {len(candidates)} candidates")

        # Compute embedding scores
        embedding_scores = self.compute_embedding_scores(query, candidates)

        # Combine scores
        scored_candidates = self.combine_scores(candidates, embedding_scores)

        # Sort by combined score
        ranked = sorted(
            scored_candidates,
            key=lambda x: x['combined_score'],
            reverse=True
        )

        # Return top-k
        top_results = ranked[:top_k]

        logger.info(
            f"Re-ranking complete. Top score: {top_results[0]['combined_score']:.4f}"
            if top_results else "No results"
        )

        return top_results

    def set_weights(self, alpha: float, beta: float):
        """
        Update scoring weights.

        Args:
            alpha: New graph score weight
            beta: New embedding score weight
        """
        total = alpha + beta
        self.alpha = alpha / total
        self.beta = beta / total

        logger.info(f"Updated weights: alpha={self.alpha:.2f}, beta={self.beta:.2f}")


class AdaptiveReranker(BottomUpReranker):
    """
    Adaptive reranker that adjusts weights based on query characteristics.

    For queries with many entities -> higher alpha (trust graph more)
    For queries with few entities -> higher beta (trust embeddings more)
    """

    def __init__(
        self,
        index_manager,
        base_alpha: float = 0.5,
        base_beta: float = 0.5,
        adaptive: bool = True
    ):
        """
        Initialize adaptive reranker.

        Args:
            index_manager: IndexManager instance
            base_alpha: Base weight for graph score
            base_beta: Base weight for embedding score
            adaptive: Enable adaptive weight adjustment
        """
        super().__init__(index_manager, base_alpha, base_beta)
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.adaptive = adaptive

    def _adjust_weights(
        self,
        num_entities: int,
        num_graph_candidates: int
    ) -> tuple:
        """
        Adjust weights based on query characteristics.

        Args:
            num_entities: Number of entities extracted from query
            num_graph_candidates: Number of candidates from graph

        Returns:
            (alpha, beta) tuple
        """
        if not self.adaptive:
            return (self.base_alpha, self.base_beta)

        # More entities + more graph candidates -> trust graph more
        if num_entities >= 3 and num_graph_candidates >= 20:
            alpha, beta = 0.7, 0.3
        elif num_entities >= 2 and num_graph_candidates >= 10:
            alpha, beta = 0.6, 0.4
        elif num_entities == 0 or num_graph_candidates < 5:
            # Few graph candidates -> rely on embeddings
            alpha, beta = 0.3, 0.7
        else:
            alpha, beta = self.base_alpha, self.base_beta

        logger.debug(
            f"Adaptive weights: alpha={alpha:.2f}, beta={beta:.2f} "
            f"(entities={num_entities}, graph_candidates={num_graph_candidates})"
        )

        return (alpha, beta)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        num_query_entities: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Re-rank with adaptive weights.

        Args:
            query: Query text
            candidates: Candidates from top-down phase
            top_k: Number of results to return
            num_query_entities: Number of entities in query (for adaptation)

        Returns:
            Re-ranked top-k candidates
        """
        # Adjust weights based on query characteristics
        if self.adaptive:
            alpha, beta = self._adjust_weights(
                num_query_entities,
                len(candidates)
            )
            self.set_weights(alpha, beta)

        # Use parent rerank method
        return super().rerank(query, candidates, top_k)


if __name__ == "__main__":
    # Example usage
    import argparse
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from index.index_utils import load_index

    parser = argparse.ArgumentParser(description="Bottom-up reranking test")
    parser.add_argument("--index-dir", type=str, required=True, help="Index directory")
    parser.add_argument("--query", type=str, required=True, help="Test query")
    parser.add_argument("--alpha", type=float, default=0.5, help="Graph score weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Embedding score weight")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive weights")
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    # Load index
    index_manager = load_index(args.index_dir)

    # Initialize reranker
    if args.adaptive:
        reranker = AdaptiveReranker(
            index_manager,
            base_alpha=args.alpha,
            base_beta=args.beta,
            adaptive=True
        )
    else:
        reranker = BottomUpReranker(
            index_manager,
            alpha=args.alpha,
            beta=args.beta
        )

    # Create mock candidates for testing
    # (normally these come from top-down phase)
    print("\nSearching index for candidates...")
    index_results = index_manager.search(args.query, top_k=50)

    mock_candidates = [
        {
            'chunk_id': r['chunk']['id'],
            'graph_score': np.random.uniform(0.3, 1.0),  # Mock graph scores
            'ctv_matches': []
        }
        for r in index_results
    ]

    print(f"Created {len(mock_candidates)} mock candidates")

    # Rerank
    print(f"\nRe-ranking with alpha={args.alpha}, beta={args.beta}...")
    results = reranker.rerank(args.query, mock_candidates, top_k=args.top_k)

    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Chunk: {result['chunk_id']}")
        print(f"   Graph score:     {result.get('graph_score', 0):.4f}")
        print(f"   Embedding score: {result.get('embedding_score', 0):.4f}")
        print(f"   Combined score:  {result['combined_score']:.4f}")
