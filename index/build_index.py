#!/usr/bin/env python3
"""
Build FAISS vector index for chunk retrieval.

Creates:
- Dense embeddings for all chunks
- FAISS index with metadata
- Mapping from index IDs to chunk IDs
"""

import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build and manage FAISS vector index."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat",
        normalize: bool = True
    ):
        """
        Initialize index builder.

        Args:
            embedding_model: HuggingFace model name
            index_type: "flat" (exact) or "ivf" (approximate)
            normalize: Normalize embeddings for cosine similarity
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index_type = index_type
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

        # Ensure embeddings are float32 and C-contiguous for FAISS compatibility
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.normalize:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)

        logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        return embeddings

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        logger.info(f"Building {self.index_type} index...")

        n_vectors, dim = embeddings.shape
        
        # Force CPU-only mode for macOS compatibility
        logger.info("Using CPU-only FAISS for macOS compatibility")

        if self.index_type == "flat":
            # Exact search (L2 or Inner Product)
            if self.normalize:
                # For normalized vectors, IP = cosine similarity
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)

        elif self.index_type == "ivf":
            # Approximate search with inverted file index
            n_centroids = min(int(np.sqrt(n_vectors)), 1000)

            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_centroids)

            logger.info(f"Training IVF index with {n_centroids} centroids...")
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add vectors to index in very small batches to avoid segmentation fault on macOS
        logger.info("Adding vectors to index in small batches (macOS workaround)...")
        batch_size = 500  # Reduced from 1000 to 500
        n_batches = (n_vectors + batch_size - 1) // batch_size
        
        for i in range(0, n_vectors, batch_size):
            end_idx = min(i + batch_size, n_vectors)
            batch = embeddings[i:end_idx].copy()  # Make a copy to ensure memory safety
            
            # Ensure batch is C-contiguous and float32
            batch = np.ascontiguousarray(batch, dtype=np.float32)
            
            try:
                index.add(batch)
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                # Try adding one by one for this batch
                logger.info("Falling back to one-by-one addition...")
                for j in range(len(batch)):
                    try:
                        single = np.ascontiguousarray(batch[j:j+1], dtype=np.float32)
                        index.add(single)
                    except Exception as e2:
                        logger.error(f"Error adding vector {i+j}: {e2}")
                        continue
            
            # Explicitly free memory
            del batch
            gc.collect()
            
            if (i // batch_size + 1) % 5 == 0 or end_idx == n_vectors:
                logger.info(f"Added batch {i//batch_size + 1}/{n_batches} ({end_idx}/{n_vectors} vectors)")

        logger.info(f"Index built with {index.ntotal} vectors")
        return index

    def save_index(
        self,
        index: faiss.Index,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        output_dir: Path
    ):
        """Save index and metadata to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = output_dir / "faiss.index"
        logger.info(f"Saving FAISS index to {index_file}")
        faiss.write_index(index, str(index_file))

        # Save chunk metadata
        metadata_file = output_dir / "metadata.pkl"
        logger.info(f"Saving metadata to {metadata_file}")

        # Create ID mapping and metadata
        id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

        metadata_obj = {
            'id_to_chunk': id_to_chunk,
            'config': metadata,
            'num_chunks': len(chunks),
            'embedding_dim': self.embedding_dim
        }

        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata_obj, f)

        # Save chunk IDs for quick lookup
        ids_file = output_dir / "chunk_ids.json"
        chunk_ids = [chunk['id'] for chunk in chunks]
        with open(ids_file, 'w') as f:
            json.dump(chunk_ids, f, indent=2)

        logger.info(f"Index and metadata saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
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
        "--index-type",
        type=str,
        choices=["flat", "ivf"],
        default="flat",
        help="Index type (flat=exact, ivf=approximate)"
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
    builder = IndexBuilder(
        embedding_model=args.model,
        index_type=args.index_type,
        normalize=not args.no_normalize
    )

    # Load chunks
    chunks = builder.load_chunks(Path(args.chunks))

    if not chunks:
        logger.error("No chunks found!")
        return

    # Compute embeddings
    embeddings = builder.embed_chunks(chunks, batch_size=args.batch_size)

    # Build index
    index = builder.build_index(embeddings)

    # Save index and metadata
    metadata = {
        'embedding_model': args.model,
        'index_type': args.index_type,
        'normalized': not args.no_normalize,
        'num_chunks': len(chunks)
    }

    builder.save_index(index, chunks, metadata, Path(args.out))

    logger.info("Index building completed successfully!")


if __name__ == "__main__":
    main()
