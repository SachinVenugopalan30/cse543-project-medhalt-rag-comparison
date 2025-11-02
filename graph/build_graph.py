#!/usr/bin/env python3
"""
Build Triple Graph Construction (TGC) from enriched chunks.

Graph structure:
- Nodes: Chunk, SourceDoc, CTV (Controlled Vocabulary)
- Edges: Chunk->SourceDoc, Chunk->CTV, SourceDoc->CTV
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx
from neo4j import GraphDatabase
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build knowledge graph from chunks."""

    def __init__(self, use_neo4j: bool = False, neo4j_uri: str = None):
        """
        Initialize graph builder.

        Args:
            use_neo4j: Use Neo4j database (vs NetworkX)
            neo4j_uri: Neo4j connection URI
        """
        self.use_neo4j = use_neo4j

        if use_neo4j:
            if not neo4j_uri:
                raise ValueError("neo4j_uri required when use_neo4j=True")

            logger.info(f"Connecting to Neo4j at {neo4j_uri}")
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=("neo4j", "medhalt2024")  # From docker-compose.yml
            )
            self._clear_neo4j()
        else:
            logger.info("Using NetworkX for graph storage")
            self.graph = nx.MultiDiGraph()

    def _clear_neo4j(self):
        """Clear existing Neo4j database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing Neo4j data")

    def add_chunk_node(self, chunk: Dict[str, Any]):
        """Add chunk node to graph."""
        if self.use_neo4j:
            self._add_chunk_neo4j(chunk)
        else:
            self._add_chunk_nx(chunk)

    def _add_chunk_nx(self, chunk: Dict[str, Any]):
        """Add chunk to NetworkX graph."""
        chunk_id = chunk['id']
        source_id = chunk['source_id']
        ctv_codes = chunk.get('ctv_codes', [])

        # Add chunk node
        self.graph.add_node(
            chunk_id,
            node_type='chunk',
            text=chunk['text'][:500],  # Store preview
            source=chunk['source'],
            title=chunk.get('title', ''),
            token_count=chunk.get('token_count', 0)
        )

        # Add source document node
        self.graph.add_node(
            f"doc:{source_id}",
            node_type='source_doc',
            source_id=source_id,
            source=chunk['source'],
            title=chunk.get('title', '')
        )

        # Add edge: chunk -> source doc
        self.graph.add_edge(
            chunk_id,
            f"doc:{source_id}",
            edge_type='from_document'
        )

        # Add CTV nodes and edges
        for ctv_code in ctv_codes:
            # Add CTV node (if not exists)
            ctv_node_id = f"ctv:{ctv_code}"
            if ctv_node_id not in self.graph:
                # Parse CTV type (e.g., "ICD10:A00" -> type="ICD10")
                ctv_type = ctv_code.split(':')[0] if ':' in ctv_code else 'UNKNOWN'

                self.graph.add_node(
                    ctv_node_id,
                    node_type='ctv',
                    ctv_code=ctv_code,
                    ctv_type=ctv_type
                )

            # Add edges: chunk -> CTV, source -> CTV
            self.graph.add_edge(
                chunk_id,
                ctv_node_id,
                edge_type='mentions_concept'
            )

            self.graph.add_edge(
                f"doc:{source_id}",
                ctv_node_id,
                edge_type='about_concept'
            )

    def _add_chunk_neo4j(self, chunk: Dict[str, Any]):
        """Add chunk to Neo4j graph."""
        chunk_id = chunk['id']
        source_id = chunk['source_id']
        ctv_codes = chunk.get('ctv_codes', [])

        with self.driver.session() as session:
            # Create chunk node
            session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.source = $source,
                    c.title = $title,
                    c.token_count = $token_count
                """,
                chunk_id=chunk_id,
                text=chunk['text'][:1000],
                source=chunk['source'],
                title=chunk.get('title', ''),
                token_count=chunk.get('token_count', 0)
            )

            # Create source doc node and relationship
            session.run(
                """
                MERGE (d:SourceDoc {id: $source_id})
                SET d.source = $source,
                    d.title = $title

                WITH d
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:FROM_DOCUMENT]->(d)
                """,
                source_id=source_id,
                source=chunk['source'],
                title=chunk.get('title', ''),
                chunk_id=chunk_id
            )

            # Create CTV nodes and relationships
            for ctv_code in ctv_codes:
                ctv_type = ctv_code.split(':')[0] if ':' in ctv_code else 'UNKNOWN'

                session.run(
                    """
                    MERGE (ctv:CTV {code: $ctv_code})
                    SET ctv.type = $ctv_type

                    WITH ctv
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (d:SourceDoc {id: $source_id})

                    MERGE (c)-[:MENTIONS_CONCEPT]->(ctv)
                    MERGE (d)-[:ABOUT_CONCEPT]->(ctv)
                    """,
                    ctv_code=ctv_code,
                    ctv_type=ctv_type,
                    chunk_id=chunk_id,
                    source_id=source_id
                )

    def build_from_chunks(self, chunks_dir: Path):
        """Build graph from all chunk files."""
        logger.info(f"Building graph from chunks in {chunks_dir}")

        chunk_count = 0
        for jsonl_file in chunks_dir.glob("*.jsonl"):
            logger.info(f"Processing {jsonl_file}")

            with open(jsonl_file, 'r') as f:
                for line in tqdm(f, desc=f"Loading {jsonl_file.name}"):
                    try:
                        chunk = json.loads(line)
                        self.add_chunk_node(chunk)
                        chunk_count += 1

                        if chunk_count % 1000 == 0:
                            logger.info(f"Processed {chunk_count} chunks")

                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")

        logger.info(f"Graph built with {chunk_count} chunks")

        if not self.use_neo4j:
            logger.info(f"NetworkX graph stats:")
            logger.info(f"  Nodes: {self.graph.number_of_nodes()}")
            logger.info(f"  Edges: {self.graph.number_of_edges()}")

    def save_networkx_graph(self, output_file: Path):
        """Save NetworkX graph to disk."""
        if self.use_neo4j:
            logger.warning("Cannot save NetworkX graph when using Neo4j")
            return

        import pickle

        logger.info(f"Saving NetworkX graph to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'wb') as f:
            pickle.dump(self.graph, f)

        logger.info(f"Graph saved to {output_file}")

    def create_neo4j_indexes(self):
        """Create indexes for Neo4j performance."""
        if not self.use_neo4j:
            return

        logger.info("Creating Neo4j indexes...")

        with self.driver.session() as session:
            # Create indexes
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:SourceDoc) ON (d.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (ctv:CTV) ON (ctv.code)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (ctv:CTV) ON (ctv.type)")

        logger.info("Indexes created")

    def close(self):
        """Close database connections."""
        if self.use_neo4j and hasattr(self, 'driver'):
            self.driver.close()


def main():
    parser = argparse.ArgumentParser(description="Build Triple Graph Construction")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Directory containing enriched chunk JSONL files"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="graph",
        help="Output directory (for NetworkX pickle)"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        help="Neo4j URI (e.g., bolt://localhost:7687)"
    )
    parser.add_argument(
        "--use-neo4j",
        action="store_true",
        help="Use Neo4j instead of NetworkX"
    )

    args = parser.parse_args()

    # Initialize builder
    builder = GraphBuilder(
        use_neo4j=args.use_neo4j or bool(args.neo4j_uri),
        neo4j_uri=args.neo4j_uri
    )

    try:
        # Build graph
        builder.build_from_chunks(Path(args.chunks))

        # Create indexes (Neo4j only)
        if args.use_neo4j or args.neo4j_uri:
            builder.create_neo4j_indexes()
        else:
            # Save NetworkX graph
            output_file = Path(args.out) / "graph.pkl"
            builder.save_networkx_graph(output_file)

        logger.info("Graph building completed successfully!")

    finally:
        builder.close()


if __name__ == "__main__":
    main()
