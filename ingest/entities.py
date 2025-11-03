#!/usr/bin/env python3
"""
Extract medical entities from chunks and map to controlled vocabularies (ICD-10, MeSH).

Uses:
- SciSpaCy for biomedical NER
- PubTator annotations (optional)
- Simple string matching to ICD/MeSH codes
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import spacy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted biomedical entity."""
    text: str
    label: str  # Disease, Drug, Gene, etc.
    start: int
    end: int
    ctv_codes: List[str] = None  # ICD/MeSH codes

    def __post_init__(self):
        if self.ctv_codes is None:
            self.ctv_codes = []


class EntityExtractor:
    """Extract and normalize biomedical entities."""

    def __init__(
        self,
        model_name: str = "en_core_sci_sm",
        icd_mapping: Dict[str, str] = None,
        mesh_mapping: Dict[str, str] = None
    ):
        logger.info(f"Loading SciSpaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(
                f"Model {model_name} not found. "
                "Install with: python -m spacy download en_core_sci_sm"
            )
            raise

        self.icd_mapping = icd_mapping or {}
        self.mesh_mapping = mesh_mapping or {}

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using SciSpaCy."""
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            )

            # Map to controlled vocabulary
            entity.ctv_codes = self._map_to_ctv(ent.text, ent.label_)
            entities.append(entity)

        return entities

    def _map_to_ctv(self, entity_text: str, entity_label: str) -> List[str]:
        """Map entity to ICD-10 or MeSH codes."""
        codes = []

        # Normalize entity text
        normalized = entity_text.lower().strip()

        # Try ICD-10 mapping (for diseases/conditions)
        # Note: Some SciSpaCy models use generic "ENTITY" label
        if entity_label in ['DISEASE', 'SYMPTOM', 'CONDITION', 'ENTITY']:
            if normalized in self.icd_mapping:
                codes.append(f"ICD10:{self.icd_mapping[normalized]}")

        # Try MeSH mapping (broader coverage)
        if normalized in self.mesh_mapping:
            codes.append(f"MESH:{self.mesh_mapping[normalized]}")

        return codes

    def load_icd10_mapping(self, icd_file: Path) -> Dict[str, str]:
        """Load ICD-10 code mappings from file."""
        logger.info(f"Loading ICD-10 mappings from {icd_file}")
        mapping = {}

        try:
            with open(icd_file, 'r') as f:
                for line in f:
                    # Expected format: code,description
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        code, desc = parts
                        mapping[desc.lower().strip()] = code.strip()

            logger.info(f"Loaded {len(mapping)} ICD-10 mappings")
            self.icd_mapping = mapping
            return mapping

        except Exception as e:
            logger.error(f"Error loading ICD-10 mappings: {e}")
            return {}

    def load_mesh_mapping(self, mesh_file: Path) -> Dict[str, str]:
        """Load MeSH code mappings from file."""
        logger.info(f"Loading MeSH mappings from {mesh_file}")
        mapping = {}

        try:
            with open(mesh_file, 'r') as f:
                for line in f:
                    # Expected format: mesh_id,term
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        mesh_id, term = parts
                        mapping[term.lower().strip()] = mesh_id.strip()

            logger.info(f"Loaded {len(mapping)} MeSH mappings")
            self.mesh_mapping = mapping
            return mapping

        except Exception as e:
            logger.error(f"Error loading MeSH mappings: {e}")
            return {}


class PubTatorAnnotator:
    """Load pre-computed PubTator annotations."""

    def __init__(self, pubtator_dir: Path):
        self.pubtator_dir = pubtator_dir
        self.annotations = {}

    def load_annotations(self):
        """Load PubTator annotations from files."""
        logger.info(f"Loading PubTator annotations from {self.pubtator_dir}")

        # PubTator files are typically tab-delimited
        # Format: PMID \t Type \t Concept_ID \t Mentions \t Resource
        annotation_files = list(self.pubtator_dir.glob("*.txt"))

        for file_path in annotation_files:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        pmid = parts[0]
                        entity_type = parts[1]
                        concept_id = parts[2]

                        if pmid not in self.annotations:
                            self.annotations[pmid] = []

                        self.annotations[pmid].append({
                            'type': entity_type,
                            'concept_id': concept_id
                        })

        logger.info(f"Loaded annotations for {len(self.annotations)} documents")

    def get_annotations(self, pmid: str) -> List[Dict[str, Any]]:
        """Get annotations for a specific PMID."""
        return self.annotations.get(pmid, [])


def enrich_chunks(
    chunks_file: Path,
    output_file: Path,
    extractor: EntityExtractor,
    pubtator: PubTatorAnnotator = None
):
    """Add entity annotations to chunks."""
    logger.info(f"Enriching chunks from {chunks_file}")

    enriched_count = 0
    entity_count = 0

    with open(output_file, 'w') as outf:
        with open(chunks_file, 'r') as inf:
            for line_num, line in enumerate(inf, 1):
                try:
                    chunk = json.loads(line)

                    # Extract entities
                    entities = extractor.extract_entities(chunk['text'])

                    # Add PubTator annotations if available
                    if pubtator and 'source_id' in chunk:
                        pubtator_annot = pubtator.get_annotations(chunk['source_id'])
                        # Merge with extracted entities (simplified)
                        chunk['pubtator_annotations'] = pubtator_annot

                    # Add entities to chunk
                    chunk['entities'] = [
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

                    chunk['ctv_codes'] = list(ctv_codes)

                    outf.write(json.dumps(chunk) + '\n')

                    enriched_count += 1
                    entity_count += len(entities)

                    if line_num % 1000 == 0:
                        logger.info(
                            f"Processed {line_num} chunks, "
                            f"found {entity_count} entities"
                        )

                except Exception as e:
                    logger.error(f"Error processing chunk on line {line_num}: {e}")

    logger.info(
        f"Enriched {enriched_count} chunks with {entity_count} entities"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract entities and map to CTVs")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Input chunks directory or JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: same as chunks with _enriched suffix)"
    )
    parser.add_argument(
        "--pubtator",
        type=str,
        help="PubTator annotations directory (optional)"
    )
    parser.add_argument(
        "--icd",
        type=str,
        help="ICD-10 mapping file (CSV format)"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        help="MeSH mapping file (CSV format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_sci_sm",
        help="SciSpaCy model name"
    )

    args = parser.parse_args()

    # Initialize entity extractor
    extractor = EntityExtractor(model_name=args.model)

    # Load CTV mappings
    if args.icd:
        extractor.load_icd10_mapping(Path(args.icd))

    if args.mesh:
        extractor.load_mesh_mapping(Path(args.mesh))

    # Load PubTator annotations
    pubtator = None
    if args.pubtator:
        pubtator = PubTatorAnnotator(Path(args.pubtator))
        pubtator.load_annotations()

    # Process chunks
    chunks_path = Path(args.chunks)

    if chunks_path.is_file():
        # Single file
        output_file = Path(args.output) if args.output else chunks_path.parent / f"{chunks_path.stem}_enriched.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        enrich_chunks(chunks_path, output_file, extractor, pubtator)
    elif chunks_path.is_dir():
        # Directory of chunk files
        output_dir = Path(args.output) if args.output else chunks_path.parent / "chunks_enriched"
        output_dir.mkdir(parents=True, exist_ok=True)

        for chunk_file in chunks_path.glob("*.jsonl"):
            output_file = output_dir / f"{chunk_file.stem}_enriched.jsonl"
            enrich_chunks(chunk_file, output_file, extractor, pubtator)
    else:
        logger.error(f"Chunks path not found: {chunks_path}")
        return

    logger.info("Entity extraction completed!")


if __name__ == "__main__":
    main()
