#!/usr/bin/env python3
"""
Document chunking for biomedical texts.

Implements:
- Static chunking (fixed token windows)
- Semantic chunking (concept-cohesive clusters)
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class Chunk:
    """Represents a document chunk."""
    id: str
    text: str
    source: str
    source_id: str  # PMID, PMC ID, etc.
    title: str
    start_char: int
    end_char: int
    token_count: int
    entities: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []


class DocumentChunker:
    """Chunk documents using static and semantic strategies."""

    def __init__(
        self,
        max_tokens: int = 1000,
        overlap_tokens: int = 100,
        use_semantic: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_semantic = use_semantic

        if use_semantic:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None

    def count_tokens(self, text: str) -> int:
        """Approximate token count (words as proxy)."""
        # In production, use tiktoken or proper tokenizer
        return len(text.split())

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return nltk.sent_tokenize(text)

    def static_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Create fixed-size chunks with overlap."""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        char_offset = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)

            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{metadata['source_id']}_chunk_{len(chunks)}"

                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=metadata['source'],
                    source_id=metadata['source_id'],
                    title=metadata.get('title', ''),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=current_tokens
                ))

                # Overlap: keep last few sentences
                overlap_sents = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap_tokens:
                        overlap_sents.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sents
                current_tokens = overlap_tokens
                char_offset += len(chunk_text)

            current_chunk.append(sent)
            current_tokens += sent_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{metadata['source_id']}_chunk_{len(chunks)}"
            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text,
                source=metadata['source'],
                source_id=metadata['source_id'],
                title=metadata.get('title', ''),
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                token_count=current_tokens
            ))

        return chunks

    def semantic_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        similarity_threshold: float = 0.3
    ) -> List[Chunk]:
        """Create concept-cohesive chunks using sentence embeddings."""
        if not self.embedder:
            logger.warning("Semantic chunking requires embedding model. Falling back to static.")
            return self.static_chunk(text, metadata)

        sentences = self.split_into_sentences(text)

        if len(sentences) <= 2:
            # Too short for clustering
            return self.static_chunk(text, metadata)

        # Compute sentence embeddings
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        embeddings = self.embedder.encode(sentences)

        # Agglomerative clustering
        n_clusters = max(1, len(sentences) // 5)  # Heuristic: ~5 sentences per cluster
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(sentences)),
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        # Group sentences by cluster
        clusters = {}
        for sent, label in zip(sentences, labels):
            clusters.setdefault(label, []).append(sent)

        # Create chunks from clusters
        chunks = []
        char_offset = 0

        for cluster_id, cluster_sents in sorted(clusters.items()):
            chunk_text = " ".join(cluster_sents)
            token_count = self.count_tokens(chunk_text)

            # Split large clusters
            if token_count > self.max_tokens:
                # Fallback to static chunking for this cluster
                cluster_text = " ".join(cluster_sents)
                sub_chunks = self.static_chunk(cluster_text, metadata)
                chunks.extend(sub_chunks)
            else:
                chunk_id = f"{metadata['source_id']}_chunk_{len(chunks)}"
                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=metadata['source'],
                    source_id=metadata['source_id'],
                    title=metadata.get('title', ''),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=token_count
                ))

            char_offset += len(chunk_text)

        return chunks

    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single document."""
        text = document.get('text', '')
        metadata = {
            'source': document.get('source', 'unknown'),
            'source_id': document.get('id', 'unknown'),
            'title': document.get('title', '')
        }

        if self.use_semantic:
            return self.semantic_chunk(text, metadata)
        else:
            return self.static_chunk(text, metadata)


def process_jsonl_file(
    input_file: Path,
    output_file: Path,
    chunker: DocumentChunker
):
    """Process a JSONL file of documents."""
    logger.info(f"Processing {input_file}")

    chunk_count = 0
    with open(output_file, 'w') as outf:
        with open(input_file, 'r') as inf:
            for line_num, line in enumerate(inf, 1):
                try:
                    doc = json.loads(line)
                    chunks = chunker.chunk_document(doc)

                    for chunk in chunks:
                        outf.write(json.dumps(asdict(chunk)) + '\n')
                        chunk_count += 1

                    if line_num % 100 == 0:
                        logger.info(f"Processed {line_num} documents, created {chunk_count} chunks")

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing document on line {line_num}: {e}")

    logger.info(f"Created {chunk_count} chunks from {line_num} documents")


def main():
    parser = argparse.ArgumentParser(description="Chunk biomedical documents")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory or JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for chunks"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens per chunk"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap tokens between chunks"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking (use static only)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for semantic chunking"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize chunker
    chunker = DocumentChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        use_semantic=not args.no_semantic,
        embedding_model=args.embedding_model
    )

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Single JSONL file
        output_file = output_dir / f"{input_path.stem}_chunks.jsonl"
        process_jsonl_file(input_path, output_file, chunker)
    elif input_path.is_dir():
        # Directory of JSONL files
        for jsonl_file in input_path.glob("*.jsonl"):
            output_file = output_dir / f"{jsonl_file.stem}_chunks.jsonl"
            process_jsonl_file(jsonl_file, output_file, chunker)
    else:
        logger.error(f"Input path not found: {input_path}")
        return

    logger.info("Chunking completed!")


if __name__ == "__main__":
    main()
