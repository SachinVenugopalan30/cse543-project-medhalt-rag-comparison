#!/usr/bin/env python3
"""
Pipeline to build the complete corpus from raw data to enriched chunks.

Orchestrates:
1. Document loading from various sources
2. Chunking
3. Entity extraction and CTV mapping
"""

import argparse
import logging
from pathlib import Path
import json
from typing import Iterator, Dict, Any
from chunker import DocumentChunker
from entities import EntityExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Build the complete corpus pipeline."""

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        chunker: DocumentChunker,
        extractor: EntityExtractor
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.chunker = chunker
        self.extractor = extractor

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_pubmed_documents(self) -> Iterator[Dict[str, Any]]:
        """Load PubMed documents from JSONL files."""
        pubmed_dir = self.raw_dir / "pubmed_baseline"

        if not pubmed_dir.exists():
            logger.warning(f"PubMed directory not found: {pubmed_dir}")
            return

        for jsonl_file in pubmed_dir.glob("*.jsonl"):
            logger.info(f"Loading {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        # Normalize document format
                        yield {
                            'id': doc.get('pmid', doc.get('id')),
                            'title': doc.get('title', ''),
                            'text': doc.get('abstract', doc.get('text', '')),
                            'source': 'pubmed'
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")

    def load_medlineplus_documents(self) -> Iterator[Dict[str, Any]]:
        """Load MedlinePlus health topics."""
        medlineplus_dir = self.raw_dir / "medlineplus"

        if not medlineplus_dir.exists():
            logger.warning(f"MedlinePlus directory not found: {medlineplus_dir}")
            return

        # MedlinePlus XML parsing would go here
        # For now, assume preprocessed JSONL
        for jsonl_file in medlineplus_dir.glob("*.jsonl"):
            logger.info(f"Loading {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        yield {
                            'id': doc.get('topic_id', doc.get('id')),
                            'title': doc.get('title', ''),
                            'text': doc.get('summary', doc.get('text', '')),
                            'source': 'medlineplus'
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")

    def process_source(
        self,
        source_name: str,
        document_loader: Iterator[Dict[str, Any]]
    ) -> int:
        """Process documents from a single source."""
        logger.info(f"Processing {source_name} documents...")

        output_file = self.output_dir / f"{source_name}_chunks.jsonl"
        chunk_count = 0
        doc_count = 0

        with open(output_file, 'w') as f:
            for doc in document_loader:
                try:
                    # Chunk document
                    chunks = self.chunker.chunk_document(doc)

                    # Extract entities and enrich each chunk
                    for chunk in chunks:
                        # Extract entities
                        entities = self.extractor.extract_entities(chunk.text)

                        # Add entities to chunk
                        chunk.entities = [
                            {
                                'text': e.text,
                                'label': e.label,
                                'start': e.start,
                                'end': e.end,
                                'ctv_codes': e.ctv_codes
                            }
                            for e in entities
                        ]

                        # Extract unique CTV codes
                        ctv_codes = set()
                        for e in entities:
                            ctv_codes.update(e.ctv_codes)

                        # Convert chunk to dict and add ctv_codes
                        chunk_dict = {
                            'id': chunk.id,
                            'text': chunk.text,
                            'source': chunk.source,
                            'source_id': chunk.source_id,
                            'title': chunk.title,
                            'start_char': chunk.start_char,
                            'end_char': chunk.end_char,
                            'token_count': chunk.token_count,
                            'entities': chunk.entities,
                            'ctv_codes': list(ctv_codes)
                        }

                        f.write(json.dumps(chunk_dict) + '\n')
                        chunk_count += 1

                    doc_count += 1

                    if doc_count % 100 == 0:
                        logger.info(
                            f"Processed {doc_count} {source_name} documents, "
                            f"created {chunk_count} chunks"
                        )

                except Exception as e:
                    logger.error(f"Error processing document: {e}")

        logger.info(
            f"Completed {source_name}: {doc_count} documents → {chunk_count} chunks"
        )
        return chunk_count

    def build_corpus(self):
        """Build the complete corpus from all sources."""
        logger.info("Starting corpus build...")

        total_chunks = 0

        # Process PubMed
        total_chunks += self.process_source(
            "pubmed",
            self.load_pubmed_documents()
        )

        # Process MedlinePlus
        total_chunks += self.process_source(
            "medlineplus",
            self.load_medlineplus_documents()
        )

        # Additional sources can be added here:
        # - PMC Open Access
        # - OpenFDA drug labels
        # - CDC guidelines

        logger.info(f"Corpus build complete! Total chunks: {total_chunks}")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build complete corpus pipeline")
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw",
        help="Raw data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chunks",
        help="Output directory for enriched chunks"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens per chunk"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking"
    )
    parser.add_argument(
        "--icd",
        type=str,
        help="ICD-10 mapping file"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        help="MeSH mapping file"
    )

    args = parser.parse_args()

    # Initialize components
    logger.info("Initializing chunker and entity extractor...")

    chunker = DocumentChunker(
        max_tokens=args.max_tokens,
        use_semantic=not args.no_semantic
    )

    extractor = EntityExtractor()

    # Load CTV mappings
    if args.icd:
        extractor.load_icd10_mapping(Path(args.icd))
    if args.mesh:
        extractor.load_mesh_mapping(Path(args.mesh))

    # Build corpus
    builder = CorpusBuilder(
        raw_dir=Path(args.raw),
        output_dir=Path(args.output),
        chunker=chunker,
        extractor=extractor
    )

    builder.build_corpus()


if __name__ == "__main__":
    main()
