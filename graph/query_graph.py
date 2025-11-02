#!/usr/bin/env python3
"""
Query operations for the knowledge graph.

Provides graph traversal and retrieval operations for U-Retrieval.
"""

import logging
from typing import List, Dict, Any, Set
from pathlib import Path
import networkx as nx
from neo4j import GraphDatabase
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphQuerier:
    """Query knowledge graph for retrieval."""

    def __init__(
        self,
        graph_path: Path = None,
        neo4j_uri: str = None,
        neo4j_auth: tuple = ("neo4j", "medhalt2024")
    ):
        """
        Initialize graph querier.

        Args:
            graph_path: Path to NetworkX pickle file
            neo4j_uri: Neo4j connection URI
            neo4j_auth: Neo4j authentication tuple
        """
        if neo4j_uri:
            logger.info(f"Connecting to Neo4j at {neo4j_uri}")
            self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            self.use_neo4j = True
        elif graph_path:
            logger.info(f"Loading NetworkX graph from {graph_path}")
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            self.use_neo4j = False
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes")
        else:
            raise ValueError("Must provide either graph_path or neo4j_uri")

    def find_ctv_nodes(self, ctv_codes: List[str]) -> List[str]:
        """
        Find CTV nodes matching given codes.

        Args:
            ctv_codes: List of CTV codes to find

        Returns:
            List of CTV node IDs
        """
        if self.use_neo4j:
            return self._find_ctv_neo4j(ctv_codes)
        else:
            return self._find_ctv_nx(ctv_codes)

    def _find_ctv_nx(self, ctv_codes: List[str]) -> List[str]:
        """Find CTV nodes in NetworkX graph."""
        ctv_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'ctv':
                if data.get('ctv_code') in ctv_codes:
                    ctv_nodes.append(node)
        return ctv_nodes

    def _find_ctv_neo4j(self, ctv_codes: List[str]) -> List[str]:
        """Find CTV nodes in Neo4j."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (ctv:CTV)
                WHERE ctv.code IN $codes
                RETURN ctv.code AS code
                """,
                codes=ctv_codes
            )
            return [record['code'] for record in result]

    def get_chunks_from_ctv(
        self,
        ctv_codes: List[str],
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get chunks connected to CTV codes via graph traversal.

        Args:
            ctv_codes: List of CTV codes
            max_depth: Maximum traversal depth

        Returns:
            List of chunk information with graph scores
        """
        if self.use_neo4j:
            return self._get_chunks_neo4j(ctv_codes, max_depth)
        else:
            return self._get_chunks_nx(ctv_codes, max_depth)

    def _get_chunks_nx(
        self,
        ctv_codes: List[str],
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Get chunks from NetworkX graph."""
        # Find CTV nodes
        ctv_nodes = self._find_ctv_nx(ctv_codes)

        if not ctv_nodes:
            logger.warning(f"No CTV nodes found for codes: {ctv_codes}")
            return []

        # Traverse to find connected chunks
        chunk_scores = {}

        for ctv_node in ctv_nodes:
            # Find chunks connected to this CTV
            # Traverse: CTV <- Chunk
            predecessors = list(self.graph.predecessors(ctv_node))

            for pred in predecessors:
                pred_data = self.graph.nodes[pred]

                if pred_data.get('node_type') == 'chunk':
                    chunk_id = pred

                    # Calculate graph score (based on path length/centrality)
                    # Simple version: count connections
                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = {
                            'chunk_id': chunk_id,
                            'score': 0,
                            'ctv_matches': []
                        }

                    chunk_scores[chunk_id]['score'] += 1.0
                    chunk_scores[chunk_id]['ctv_matches'].append(
                        self.graph.nodes[ctv_node].get('ctv_code')
                    )

        # Convert to list and sort by score
        chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return chunks

    def _get_chunks_neo4j(
        self,
        ctv_codes: List[str],
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Get chunks from Neo4j graph."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (ctv:CTV)<-[:MENTIONS_CONCEPT]-(c:Chunk)
                WHERE ctv.code IN $codes
                WITH c, collect(DISTINCT ctv.code) AS matched_ctvs, count(DISTINCT ctv) AS score
                RETURN c.id AS chunk_id, score, matched_ctvs
                ORDER BY score DESC
                """,
                codes=ctv_codes
            )

            chunks = [
                {
                    'chunk_id': record['chunk_id'],
                    'score': float(record['score']),
                    'ctv_matches': record['matched_ctvs']
                }
                for record in result
            ]

        return chunks

    def get_source_docs_from_ctv(
        self,
        ctv_codes: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get source documents related to CTV codes.

        Args:
            ctv_codes: List of CTV codes

        Returns:
            List of source document information
        """
        if self.use_neo4j:
            return self._get_sources_neo4j(ctv_codes)
        else:
            return self._get_sources_nx(ctv_codes)

    def _get_sources_nx(self, ctv_codes: List[str]) -> List[Dict[str, Any]]:
        """Get source docs from NetworkX graph."""
        ctv_nodes = self._find_ctv_nx(ctv_codes)
        source_docs = {}

        for ctv_node in ctv_nodes:
            predecessors = list(self.graph.predecessors(ctv_node))

            for pred in predecessors:
                pred_data = self.graph.nodes[pred]

                if pred_data.get('node_type') == 'source_doc':
                    doc_id = pred_data.get('source_id')

                    if doc_id not in source_docs:
                        source_docs[doc_id] = {
                            'source_id': doc_id,
                            'source': pred_data.get('source'),
                            'title': pred_data.get('title', ''),
                            'score': 0
                        }

                    source_docs[doc_id]['score'] += 1.0

        return sorted(
            source_docs.values(),
            key=lambda x: x['score'],
            reverse=True
        )

    def _get_sources_neo4j(self, ctv_codes: List[str]) -> List[Dict[str, Any]]:
        """Get source docs from Neo4j."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (ctv:CTV)<-[:ABOUT_CONCEPT]-(d:SourceDoc)
                WHERE ctv.code IN $codes
                WITH d, count(DISTINCT ctv) AS score
                RETURN d.id AS source_id, d.source AS source,
                       d.title AS title, score
                ORDER BY score DESC
                """,
                codes=ctv_codes
            )

            return [
                {
                    'source_id': record['source_id'],
                    'source': record['source'],
                    'title': record['title'],
                    'score': float(record['score'])
                }
                for record in result
            ]

    def get_related_ctvs(
        self,
        ctv_codes: List[str],
        max_related: int = 10
    ) -> List[str]:
        """
        Find related CTV codes (co-occurring in same documents).

        Args:
            ctv_codes: Input CTV codes
            max_related: Maximum number of related codes to return

        Returns:
            List of related CTV codes
        """
        if self.use_neo4j:
            return self._get_related_ctvs_neo4j(ctv_codes, max_related)
        else:
            return self._get_related_ctvs_nx(ctv_codes, max_related)

    def _get_related_ctvs_nx(
        self,
        ctv_codes: List[str],
        max_related: int
    ) -> List[str]:
        """Find related CTVs in NetworkX graph."""
        ctv_nodes = self._find_ctv_nx(ctv_codes)
        related_ctvs = {}

        for ctv_node in ctv_nodes:
            # Find source docs for this CTV
            predecessors = list(self.graph.predecessors(ctv_node))

            for pred in predecessors:
                if self.graph.nodes[pred].get('node_type') == 'source_doc':
                    # Find other CTVs from this source
                    successors = list(self.graph.successors(pred))

                    for succ in successors:
                        succ_data = self.graph.nodes[succ]

                        if (succ_data.get('node_type') == 'ctv' and
                            succ != ctv_node):
                            related_code = succ_data.get('ctv_code')

                            if related_code not in ctv_codes:
                                related_ctvs[related_code] = \
                                    related_ctvs.get(related_code, 0) + 1

        # Sort and return top related
        sorted_related = sorted(
            related_ctvs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [code for code, _ in sorted_related[:max_related]]

    def _get_related_ctvs_neo4j(
        self,
        ctv_codes: List[str],
        max_related: int
    ) -> List[str]:
        """Find related CTVs in Neo4j."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (ctv1:CTV)<-[:ABOUT_CONCEPT]-(d:SourceDoc)-[:ABOUT_CONCEPT]->(ctv2:CTV)
                WHERE ctv1.code IN $codes AND NOT ctv2.code IN $codes
                WITH ctv2.code AS related_code, count(DISTINCT d) AS co_occurrence
                RETURN related_code
                ORDER BY co_occurrence DESC
                LIMIT $max_related
                """,
                codes=ctv_codes,
                max_related=max_related
            )

            return [record['related_code'] for record in result]

    def close(self):
        """Close database connections."""
        if self.use_neo4j and hasattr(self, 'driver'):
            self.driver.close()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Query knowledge graph")
    parser.add_argument("--graph", type=str, help="NetworkX graph pickle file")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI")
    parser.add_argument("--ctv", type=str, nargs='+', help="CTV codes to query")

    args = parser.parse_args()

    # Initialize querier
    querier = GraphQuerier(
        graph_path=Path(args.graph) if args.graph else None,
        neo4j_uri=args.neo4j_uri
    )

    try:
        if args.ctv:
            print(f"\nQuerying for CTV codes: {args.ctv}")

            # Get related chunks
            chunks = querier.get_chunks_from_ctv(args.ctv)
            print(f"\nFound {len(chunks)} related chunks:")
            for i, chunk in enumerate(chunks[:5], 1):
                print(f"{i}. {chunk['chunk_id']} (score: {chunk['score']:.2f})")
                print(f"   Matched CTVs: {', '.join(chunk['ctv_matches'])}")

            # Get related CTVs
            related = querier.get_related_ctvs(args.ctv)
            print(f"\nRelated CTV codes: {', '.join(related[:10])}")

    finally:
        querier.close()
