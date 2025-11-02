#!/usr/bin/env python3
"""
Build NumPy-based vector index as fallback for FAISS issues on macOS.

Creates:
- Dense embeddings for all chunks
- NumPy array index with metadata (simpler, no FAISS)
- Mapping from index IDs to chunk IDs

This is a fallback option if FAISS continues to crash on macOS.
Performance is slightly slower but more stable.
"""

import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyIndexBuilder:
    """Build and manage NumPy-based vector index (FAISS alternative)."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True
    ):
        """
        Initialize index builder.

        Args:
            embedding_model: HuggingFace model name
            normalize: Normalize embeddings for cosine similarity
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.normalize = normalize

        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def load_chunks(self, chunks_dir: Path) -> List[Dict[str, Any]]:
        """Load all chunks from JSONL files."""
        logger.info(f"Loading chunks from {chunks_dir}")

        chunks = []
        for jsonl_file in chunks_dir.glob("*.jsonl"):
            logger.info(f"Reading {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        chunk = json.loads(line)
                        chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")

        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> np.ndarray:
        """Compute embeddings for all chunks."""
        logger.info(f"Computing embeddings for {len(chunks)} chunks...")

        texts = [chunk['text'] for chunk in chunks]

        # Encode in batches with progress bar
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        if self.normalize:
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        return embeddings

    def save_index(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Save NumPy index and metadata to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings as NumPy array
        embeddings_file = output_dir / "embeddings.npy"
        logger.info(f"Saving embeddings to {embeddings_file}")
        np.save(embeddings_file, embeddings)

        # Save chunk metadata
        metadata_file = output_dir / "metadata.pkl"
        logger.info(f"Saving metadata to {metadata_file}")

        # Create ID mapping and metadata
        id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

        metadata_obj = {
            'id_to_chunk': id_to_chunk,
            'config': metadata,
            'num_chunks': len(chunks),
            'embedding_dim': self.embedding_dim,
            'index_type': 'numpy'  # Mark as NumPy index
        }

        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata_obj, f)

        # Save chunk IDs for quick lookup
        ids_file = output_dir / "chunk_ids.json"
        chunk_ids = [chunk['id'] for chunk in chunks]
        with open(ids_file, 'w') as f:
            json.dump(chunk_ids, f, indent=2)

        logger.info(f"NumPy index and metadata saved to {output_dir}")
        logger.info(f"Total size: {embeddings.nbytes / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Build NumPy vector index (FAISS alternative)")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Directory containing chunk JSONL files"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="index",
        help="Output directory for index"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize embeddings"
    )

    args = parser.parse_args()

    # Initialize builder
    builder = NumpyIndexBuilder(
        embedding_model=args.model,
        normalize=not args.no_normalize
    )

    # Load chunks
    chunks = builder.load_chunks(Path(args.chunks))

    if not chunks:
        logger.error("No chunks found!")
        return

    # Compute embeddings
    embeddings = builder.embed_chunks(chunks, batch_size=args.batch_size)

    # Save index and metadata
    metadata = {
        'embedding_model': args.model,
        'index_type': 'numpy',
        'normalized': not args.no_normalize,
        'num_chunks': len(chunks)
    }

    builder.save_index(embeddings, chunks, metadata, Path(args.out))

    logger.info("NumPy index building completed successfully!")
    logger.info("")
    logger.info("NOTE: This is a NumPy-based index (FAISS alternative).")
    logger.info("The retrieval code will need to use cosine similarity search")
    logger.info("instead of FAISS index.search(). This is slower but more stable on macOS.")


if __name__ == "__main__":
    main()
