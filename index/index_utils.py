#!/usr/bin/env python3
"""
Utilities for working with FAISS index.

Provides:
- Index loading
- Query interface
- Metadata lookup
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexManager:
    """Manage index for retrieval (supports both FAISS and NumPy)."""

    def __init__(self, index_dir: Path):
        """
        Load index and metadata from directory.

        Args:
            index_dir: Directory containing index files and metadata.pkl
        """
        self.index_dir = Path(index_dir)

        # Load metadata first to determine index type
        metadata_file = self.index_dir / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        self.id_to_chunk = self.metadata['id_to_chunk']
        self.config = self.metadata['config']
        self.index_type = self.metadata.get('index_type', 'faiss')

        # Load appropriate index type
        if self.index_type == 'numpy':
            # Load NumPy embeddings
            embeddings_file = self.index_dir / "embeddings.npy"
            if not embeddings_file.exists():
                raise FileNotFoundError(f"NumPy embeddings not found: {embeddings_file}")
            
            logger.info(f"Loading NumPy index from {embeddings_file}")
            self.embeddings = np.load(embeddings_file)
            self.index = None  # No FAISS index
            logger.info(f"NumPy index loaded: {self.embeddings.shape[0]} vectors")
        else:
            # Load FAISS index
            index_file = self.index_dir / "faiss.index"
            if not index_file.exists():
                raise FileNotFoundError(f"Index file not found: {index_file}")

            logger.info(f"Loading FAISS index from {index_file}")
            self.index = faiss.read_index(str(index_file))
            self.embeddings = None  # No NumPy embeddings
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

        # Load embedding model
        embedding_model = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        self.normalize = self.config.get('normalized', True)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_ctv: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search index for similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return
            filter_ctv: Optional CTV codes to filter by

        Returns:
            List of chunk results with scores
        """
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)

        if self.normalize:
            # Normalize query embedding
            norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding / (norms + 1e-8)

        # Search index based on type
        if filter_ctv:
            # With filtering (slower, requires scanning)
            results = self._search_with_filter(query_embedding, top_k, filter_ctv)
        else:
            if self.index_type == 'numpy':
                # NumPy-based cosine similarity search
                distances, indices = self._numpy_search(query_embedding, top_k)
            else:
                # FAISS search
                distances, indices = self.index.search(query_embedding, top_k)
            
            results = self._format_results(distances[0], indices[0])

        return results
    
    def _numpy_search(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform cosine similarity search using NumPy.
        
        Args:
            query_embedding: Query embedding (1, dim)
            top_k: Number of results
            
        Returns:
            distances, indices (same format as FAISS)
        """
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices (argsort in descending order)
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Convert similarities to distances (1 - similarity for consistency with FAISS)
        # Note: For normalized vectors, FAISS IP returns similarity, so we keep as-is
        top_distances = similarities[top_indices]
        
        # Return in same format as FAISS (batch dimension)
        return top_distances.reshape(1, -1), top_indices.reshape(1, -1)

    def _search_with_filter(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_ctv: List[str]
    ) -> List[Dict[str, Any]]:
        """Search with CTV code filtering."""
        # Retrieve more candidates for filtering
        total_vectors = self.embeddings.shape[0] if self.index_type == 'numpy' else self.index.ntotal
        k_candidates = min(top_k * 10, total_vectors)
        
        if self.index_type == 'numpy':
            distances, indices = self._numpy_search(query_embedding, k_candidates)
        else:
            distances, indices = self.index.search(query_embedding, k_candidates)

        # Filter by CTV codes
        filtered_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue

            chunk = self.id_to_chunk.get(idx)
            if chunk and self._matches_ctv_filter(chunk, filter_ctv):
                filtered_results.append({
                    'chunk': chunk,
                    'score': float(dist),
                    'index_id': int(idx)
                })

                if len(filtered_results) >= top_k:
                    break

        return filtered_results

    def _matches_ctv_filter(
        self,
        chunk: Dict[str, Any],
        filter_ctv: List[str]
    ) -> bool:
        """Check if chunk matches CTV filter."""
        chunk_ctvs = set(chunk.get('ctv_codes', []))
        filter_ctvs = set(filter_ctv)
        return bool(chunk_ctvs & filter_ctvs)  # Any overlap

    def _format_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Format search results."""
        results = []

        for dist, idx in zip(distances, indices):
            if idx == -1:  # Invalid index
                continue

            chunk = self.id_to_chunk.get(idx)
            if chunk:
                results.append({
                    'chunk': chunk,
                    'score': float(dist),
                    'index_id': int(idx)
                })

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve chunk by its ID."""
        for chunk in self.id_to_chunk.values():
            if chunk['id'] == chunk_id:
                return chunk
        return None

    def get_chunks_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific source document."""
        return [
            chunk for chunk in self.id_to_chunk.values()
            if chunk.get('source_id') == source_id
        ]

    def get_chunks_by_ctv(self, ctv_code: str) -> List[Dict[str, Any]]:
        """Get all chunks containing a specific CTV code."""
        return [
            chunk for chunk in self.id_to_chunk.values()
            if ctv_code in chunk.get('ctv_codes', [])
        ]

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple queries in batch.

        Args:
            queries: List of query texts
            top_k: Number of results per query

        Returns:
            List of result lists (one per query)
        """
        # Encode all queries
        query_embeddings = self.encoder.encode(queries, convert_to_numpy=True)

        if self.normalize:
            # Normalize query embeddings
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / (norms + 1e-8)

        # Batch search based on index type
        if self.index_type == 'numpy':
            # Search each query individually for NumPy
            all_results = []
            for query_emb in query_embeddings:
                distances, indices = self._numpy_search(query_emb.reshape(1, -1), top_k)
                results = self._format_results(distances[0], indices[0])
                all_results.append(results)
        else:
            # FAISS batch search
            distances, indices = self.index.search(query_embeddings, top_k)
            
            # Format results for each query
            all_results = []
            for query_distances, query_indices in zip(distances, indices):
                results = self._format_results(query_distances, query_indices)
                all_results.append(results)

        return all_results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_vectors = self.embeddings.shape[0] if self.index_type == 'numpy' else self.index.ntotal
        
        return {
            'total_vectors': total_vectors,
            'embedding_dim': self.metadata['embedding_dim'],
            'num_chunks': self.metadata['num_chunks'],
            'index_type': self.index_type,
            'normalized': self.normalize,
            'embedding_model': self.config.get('embedding_model')
        }


def load_index(index_dir: str) -> IndexManager:
    """Convenience function to load index."""
    return IndexManager(Path(index_dir))


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test index utilities")
    parser.add_argument("--index-dir", type=str, default="index", help="Index directory")
    parser.add_argument("--query", type=str, help="Test query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    # Load index
    manager = load_index(args.index_dir)

    # Show stats
    stats = manager.get_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test query
    if args.query:
        print(f"\nSearching for: {args.query}")
        results = manager.search(args.query, top_k=args.top_k)

        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            score = result['score']
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   ID: {chunk['id']}")
            print(f"   Source: {chunk['source']} ({chunk['source_id']})")
            print(f"   Text: {chunk['text'][:200]}...")
            if chunk.get('ctv_codes'):
                print(f"   CTV Codes: {', '.join(chunk['ctv_codes'][:5])}")
