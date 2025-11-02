#!/usr/bin/env python3
"""
Top-down retrieval: Graph-based candidate selection using CTV mapping.

Implements the first phase of U-Retrieval:
1. Extract entities from query
2. Map to CTV codes
3. Traverse graph to find candidate chunks
"""

import logging
from typing import List, Dict, Any, Set
from pathlib import Path
import spacy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopDownRetriever:
    """Top-down graph-based candidate selection."""

    def __init__(
        self,
        graph_querier,
        entity_model: str = "en_core_sci_sm",
        icd_mapping: Dict[str, str] = None,
        mesh_mapping: Dict[str, str] = None,
        expansion_factor: int = 2
    ):
        """
        Initialize top-down retriever.

        Args:
            graph_querier: GraphQuerier instance
            entity_model: SciSpaCy model for entity extraction
            icd_mapping: Entity text -> ICD code mapping
            mesh_mapping: Entity text -> MeSH code mapping
            expansion_factor: Multiplier for query expansion via related CTVs
        """
        self.graph_querier = graph_querier

        # Load NER model
        logger.info(f"Loading entity model: {entity_model}")
        try:
            self.nlp = spacy.load(entity_model)
        except OSError:
            logger.error(
                f"Model {entity_model} not found. "
                "Install with: python -m spacy download en_core_sci_sm"
            )
            raise

        self.icd_mapping = icd_mapping or {}
        self.mesh_mapping = mesh_mapping or {}
        self.expansion_factor = expansion_factor

    def extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from query.

        Args:
            query: Query text

        Returns:
            List of entity dictionaries
        """
        doc = self.nlp(query)

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        logger.debug(f"Extracted {len(entities)} entities from query")
        return entities

    def map_entities_to_ctv(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Map extracted entities to CTV codes.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of CTV codes
        """
        ctv_codes = []

        for entity in entities:
            text = entity['text'].lower().strip()
            label = entity['label']

            # Try ICD-10 mapping (diseases/conditions)
            if label in ['DISEASE', 'SYMPTOM', 'CONDITION']:
                if text in self.icd_mapping:
                    ctv_codes.append(f"ICD10:{self.icd_mapping[text]}")

            # Try MeSH mapping (broader coverage)
            if text in self.mesh_mapping:
                ctv_codes.append(f"MESH:{self.mesh_mapping[text]}")

        # Remove duplicates while preserving order
        seen = set()
        unique_ctvs = []
        for ctv in ctv_codes:
            if ctv not in seen:
                seen.add(ctv)
                unique_ctvs.append(ctv)

        logger.debug(f"Mapped to {len(unique_ctvs)} CTV codes")
        return unique_ctvs

    def expand_ctvs(
        self,
        ctv_codes: List[str],
        max_expansion: int = 10
    ) -> List[str]:
        """
        Expand CTV codes using graph co-occurrence.

        Args:
            ctv_codes: Initial CTV codes
            max_expansion: Maximum number of related CTVs to add

        Returns:
            Expanded list of CTV codes
        """
        if not ctv_codes:
            return []

        # Get related CTVs from graph
        related_ctvs = self.graph_querier.get_related_ctvs(
            ctv_codes,
            max_related=max_expansion
        )

        # Combine original and related
        expanded = ctv_codes + related_ctvs

        logger.debug(
            f"Expanded {len(ctv_codes)} CTVs to {len(expanded)} "
            f"(added {len(related_ctvs)} related)"
        )

        return expanded

    def retrieve_candidates(
        self,
        query: str,
        expand_query: bool = True,
        max_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate chunks using top-down graph traversal.

        Args:
            query: Query text
            expand_query: Whether to expand with related CTVs
            max_candidates: Maximum number of candidates

        Returns:
            List of candidate chunks with graph scores
        """
        logger.info(f"Top-down retrieval for: {query}")

        # Step 1: Extract entities
        entities = self.extract_query_entities(query)

        if not entities:
            logger.warning("No entities extracted from query")
            return []

        logger.info(f"Extracted entities: {[e['text'] for e in entities]}")

        # Step 2: Map to CTV codes
        ctv_codes = self.map_entities_to_ctv(entities)

        if not ctv_codes:
            logger.warning("No CTV codes mapped from entities")
            return []

        logger.info(f"Mapped CTV codes: {ctv_codes}")

        # Step 3: Expand CTVs (optional)
        if expand_query:
            ctv_codes = self.expand_ctvs(
                ctv_codes,
                max_expansion=max_candidates // self.expansion_factor
            )

        # Step 4: Traverse graph to get candidate chunks
        candidates = self.graph_querier.get_chunks_from_ctv(
            ctv_codes,
            max_depth=2
        )

        # Normalize scores (0-1 range based on max score)
        if candidates:
            max_score = max(c['score'] for c in candidates)
            if max_score > 0:
                for candidate in candidates:
                    candidate['graph_score'] = candidate['score'] / max_score
            else:
                for candidate in candidates:
                    candidate['graph_score'] = 0.0

        # Limit candidates
        candidates = candidates[:max_candidates]

        logger.info(
            f"Retrieved {len(candidates)} candidates from graph "
            f"(CTVs: {len(ctv_codes)})"
        )

        return candidates

    def load_ctv_mappings(
        self,
        icd_file: Path = None,
        mesh_file: Path = None
    ):
        """
        Load CTV mapping files.

        Args:
            icd_file: ICD-10 mapping CSV
            mesh_file: MeSH mapping CSV
        """
        if icd_file and icd_file.exists():
            logger.info(f"Loading ICD-10 mappings from {icd_file}")
            self.icd_mapping = self._load_mapping_file(icd_file)
            logger.info(f"Loaded {len(self.icd_mapping)} ICD-10 mappings")

        if mesh_file and mesh_file.exists():
            logger.info(f"Loading MeSH mappings from {mesh_file}")
            self.mesh_mapping = self._load_mapping_file(mesh_file)
            logger.info(f"Loaded {len(self.mesh_mapping)} MeSH mappings")

    def _load_mapping_file(self, file_path: Path) -> Dict[str, str]:
        """Load CTV mapping from CSV file."""
        mapping = {}

        with open(file_path, 'r') as f:
            for line in f:
                # Expected format: code,description
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    code, desc = parts
                    mapping[desc.lower().strip()] = code.strip()

        return mapping


if __name__ == "__main__":
    # Example usage
    import argparse
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from graph.query_graph import GraphQuerier

    parser = argparse.ArgumentParser(description="Top-down retrieval test")
    parser.add_argument("--graph", type=str, help="NetworkX graph pickle")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI")
    parser.add_argument("--query", type=str, required=True, help="Test query")
    parser.add_argument("--icd", type=str, help="ICD-10 mapping file")
    parser.add_argument("--mesh", type=str, help="MeSH mapping file")
    parser.add_argument("--max-candidates", type=int, default=50)

    args = parser.parse_args()

    # Initialize graph querier
    querier = GraphQuerier(
        graph_path=Path(args.graph) if args.graph else None,
        neo4j_uri=args.neo4j_uri
    )

    # Initialize top-down retriever
    retriever = TopDownRetriever(graph_querier=querier)

    # Load CTV mappings
    if args.icd or args.mesh:
        retriever.load_ctv_mappings(
            icd_file=Path(args.icd) if args.icd else None,
            mesh_file=Path(args.mesh) if args.mesh else None
        )

    try:
        # Retrieve candidates
        candidates = retriever.retrieve_candidates(
            args.query,
            max_candidates=args.max_candidates
        )

        print(f"\nRetrieved {len(candidates)} candidates:")
        for i, cand in enumerate(candidates[:10], 1):
            print(f"\n{i}. Chunk: {cand['chunk_id']}")
            print(f"   Graph score: {cand['graph_score']:.4f}")
            print(f"   Matched CTVs: {', '.join(cand.get('ctv_matches', []))}")

    finally:
        querier.close()
