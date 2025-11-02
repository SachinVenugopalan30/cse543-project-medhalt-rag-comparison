#!/usr/bin/env python3
"""
Utility functions for graph operations.

Provides:
- Graph statistics
- Visualization helpers
- Export utilities
"""

import logging
from typing import Dict, Any
from pathlib import Path
import networkx as nx
from neo4j import GraphDatabase
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_networkx_stats(graph: nx.Graph) -> Dict[str, Any]:
    """
    Get statistics for NetworkX graph.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary of graph statistics
    """
    # Count nodes by type
    node_types = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    # Count edges by type
    edge_types = {}
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    stats = {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
        'is_directed': graph.is_directed(),
        'is_multigraph': graph.is_multigraph()
    }

    # Calculate degree statistics for chunks
    chunk_degrees = [
        graph.degree(node) for node, data in graph.nodes(data=True)
        if data.get('node_type') == 'chunk'
    ]

    if chunk_degrees:
        stats['chunk_degree'] = {
            'mean': sum(chunk_degrees) / len(chunk_degrees),
            'min': min(chunk_degrees),
            'max': max(chunk_degrees)
        }

    return stats


def get_neo4j_stats(
    neo4j_uri: str,
    auth: tuple = ("neo4j", "medhalt2024")
) -> Dict[str, Any]:
    """
    Get statistics for Neo4j graph.

    Args:
        neo4j_uri: Neo4j connection URI
        auth: Authentication tuple

    Returns:
        Dictionary of graph statistics
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=auth)

    try:
        with driver.session() as session:
            # Count nodes by label
            node_counts = {}
            for label in ['Chunk', 'SourceDoc', 'CTV']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                node_counts[label.lower()] = result.single()['count']

            # Count relationships by type
            rel_counts = {}
            for rel_type in ['FROM_DOCUMENT', 'MENTIONS_CONCEPT', 'ABOUT_CONCEPT']:
                result = session.run(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
                )
                rel_counts[rel_type.lower()] = result.single()['count']

            # Get degree statistics for chunks
            result = session.run(
                """
                MATCH (c:Chunk)
                RETURN avg(size((c)-->())) AS avg_degree,
                       min(size((c)-->())) AS min_degree,
                       max(size((c)-->())) AS max_degree
                """
            )
            record = result.single()

            stats = {
                'total_nodes': sum(node_counts.values()),
                'total_edges': sum(rel_counts.values()),
                'node_types': node_counts,
                'edge_types': rel_counts,
                'chunk_degree': {
                    'mean': float(record['avg_degree'] or 0),
                    'min': int(record['min_degree'] or 0),
                    'max': int(record['max_degree'] or 0)
                }
            }

            return stats

    finally:
        driver.close()


def export_graph_sample_to_json(
    graph: nx.Graph,
    output_file: Path,
    max_nodes: int = 100
):
    """
    Export a sample of the graph to JSON for visualization.

    Args:
        graph: NetworkX graph
        output_file: Output JSON file path
        max_nodes: Maximum number of nodes to export
    """
    # Sample nodes
    all_nodes = list(graph.nodes())[:max_nodes]

    # Create subgraph
    subgraph = graph.subgraph(all_nodes)

    # Convert to JSON format
    graph_data = {
        'nodes': [
            {
                'id': node,
                **data
            }
            for node, data in subgraph.nodes(data=True)
        ],
        'edges': [
            {
                'source': u,
                'target': v,
                **data
            }
            for u, v, data in subgraph.edges(data=True)
        ]
    }

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)

    logger.info(f"Exported graph sample to {output_file}")


def find_most_connected_ctvs(
    graph: nx.Graph,
    top_k: int = 20
) -> list:
    """
    Find most connected CTV nodes (high degree centrality).

    Args:
        graph: NetworkX graph
        top_k: Number of top CTVs to return

    Returns:
        List of (ctv_node, degree) tuples
    """
    # Get CTV nodes
    ctv_nodes = [
        node for node, data in graph.nodes(data=True)
        if data.get('node_type') == 'ctv'
    ]

    # Calculate degrees
    ctv_degrees = [(node, graph.degree(node)) for node in ctv_nodes]

    # Sort and return top k
    ctv_degrees.sort(key=lambda x: x[1], reverse=True)

    return ctv_degrees[:top_k]


def find_chunks_by_source(
    graph: nx.Graph,
    source_name: str
) -> list:
    """
    Find all chunks from a specific source.

    Args:
        graph: NetworkX graph
        source_name: Source name (e.g., 'pubmed', 'medlineplus')

    Returns:
        List of chunk node IDs
    """
    chunks = [
        node for node, data in graph.nodes(data=True)
        if data.get('node_type') == 'chunk' and
           data.get('source') == source_name
    ]

    return chunks


def validate_graph_structure(graph: nx.Graph) -> Dict[str, Any]:
    """
    Validate graph structure and identify potential issues.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    # Check for disconnected components
    if not nx.is_directed(graph):
        num_components = nx.number_connected_components(graph.to_undirected())
        if num_components > 1:
            warnings.append(f"Graph has {num_components} disconnected components")

    # Check for orphaned nodes (degree 0)
    orphaned = [node for node in graph.nodes() if graph.degree(node) == 0]
    if orphaned:
        warnings.append(f"Found {len(orphaned)} orphaned nodes with no connections")

    # Check for chunks without CTV codes
    chunks_without_ctv = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'chunk':
            # Check if chunk has any outgoing edges to CTV nodes
            has_ctv = any(
                graph.nodes[neighbor].get('node_type') == 'ctv'
                for neighbor in graph.successors(node)
            )
            if not has_ctv:
                chunks_without_ctv.append(node)

    if chunks_without_ctv:
        warnings.append(
            f"Found {len(chunks_without_ctv)} chunks without CTV annotations"
        )

    # Check for source docs without chunks
    sources_without_chunks = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'source_doc':
            has_chunks = any(
                graph.nodes[neighbor].get('node_type') == 'chunk'
                for neighbor in graph.predecessors(node)
            )
            if not has_chunks:
                sources_without_chunks.append(node)

    if sources_without_chunks:
        warnings.append(
            f"Found {len(sources_without_chunks)} source docs without chunks"
        )

    validation_result = {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': get_networkx_stats(graph)
    }

    return validation_result


def print_graph_summary(stats: Dict[str, Any]):
    """
    Print a formatted graph summary.

    Args:
        stats: Graph statistics dictionary
    """
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*60)

    print(f"\nTotal Nodes: {stats['total_nodes']:,}")
    print(f"Total Edges: {stats['total_edges']:,}")

    print("\nNode Types:")
    for node_type, count in stats.get('node_types', {}).items():
        print(f"  {node_type:15s}: {count:,}")

    print("\nEdge Types:")
    for edge_type, count in stats.get('edge_types', {}).items():
        print(f"  {edge_type:20s}: {count:,}")

    if 'chunk_degree' in stats:
        print("\nChunk Connectivity:")
        chunk_deg = stats['chunk_degree']
        print(f"  Average degree: {chunk_deg['mean']:.2f}")
        print(f"  Min degree:     {chunk_deg['min']}")
        print(f"  Max degree:     {chunk_deg['max']}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Example: validate and print stats for a graph
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Graph utilities")
    parser.add_argument("--graph", type=str, help="NetworkX graph pickle file")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI")
    parser.add_argument("--validate", action="store_true", help="Validate graph structure")
    parser.add_argument("--export-sample", type=str, help="Export sample to JSON file")

    args = parser.parse_args()

    if args.graph:
        # Load NetworkX graph
        with open(args.graph, 'rb') as f:
            graph = pickle.load(f)

        stats = get_networkx_stats(graph)
        print_graph_summary(stats)

        if args.validate:
            print("\nValidating graph structure...")
            validation = validate_graph_structure(graph)

            if validation['valid']:
                print("✓ Graph structure is valid")
            else:
                print("✗ Graph has issues:")
                for issue in validation['issues']:
                    print(f"  - {issue}")

            if validation['warnings']:
                print("\nWarnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")

        if args.export_sample:
            export_graph_sample_to_json(graph, Path(args.export_sample))

    elif args.neo4j_uri:
        # Get Neo4j stats
        stats = get_neo4j_stats(args.neo4j_uri)
        print_graph_summary(stats)
