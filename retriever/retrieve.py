#!/usr/bin/env python3
"""
U-Retrieval orchestration: Combine top-down and bottom-up phases.

Complete retrieval pipeline that:
1. Uses graph to find candidate chunks (top-down)
2. Re-ranks using vector similarity (bottom-up)
3. Returns final ranked results with full chunk data
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class URetriever:
    """Unified retrieval combining graph and vector approaches."""

    def __init__(
        self,
        graph_querier,
        index_manager,
        top_down_retriever,
        bottom_up_reranker,
        fallback_to_vector: bool = True
    ):
        """
        Initialize U-Retriever.

        Args:
            graph_querier: GraphQuerier instance
            index_manager: IndexManager instance
            top_down_retriever: TopDownRetriever instance
            bottom_up_reranker: BottomUpReranker instance
            fallback_to_vector: Use vector-only if graph returns no results
        """
        self.graph_querier = graph_querier
        self.index_manager = index_manager
        self.top_down = top_down_retriever
        self.bottom_up = bottom_up_reranker
        self.fallback_to_vector = fallback_to_vector

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using U-Retrieval.

        Args:
            query: Query text
            top_k: Number of final results to return
            max_candidates: Maximum candidates from graph phase

        Returns:
            List of retrieved chunks with scores and metadata
        """
        logger.info(f"U-Retrieval for query: {query[:100]}...")

        # Phase 1: Top-down graph retrieval
        graph_candidates = self.top_down.retrieve_candidates(
            query,
            expand_query=True,
            max_candidates=max_candidates
        )

        # Check if we got graph results
        if not graph_candidates:
            logger.warning("No candidates from graph phase")

            if self.fallback_to_vector:
                logger.info("Falling back to vector-only retrieval")
                return self._vector_only_retrieve(query, top_k)
            else:
                return []

        # Phase 2: Bottom-up vector re-ranking
        num_entities = len(self.top_down.extract_query_entities(query))

        # Use adaptive reranking if available
        if hasattr(self.bottom_up, 'num_query_entities'):
            ranked_results = self.bottom_up.rerank(
                query,
                graph_candidates,
                top_k=top_k,
                num_query_entities=num_entities
            )
        else:
            ranked_results = self.bottom_up.rerank(
                query,
                graph_candidates,
                top_k=top_k
            )

        # Enrich results with full chunk data
        enriched_results = self._enrich_results(ranked_results)

        logger.info(f"Retrieved {len(enriched_results)} results")
        return enriched_results

    def _vector_only_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback to pure vector search.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of results from vector search only
        """
        results = self.index_manager.search(query, top_k=top_k)

        # Format to match U-Retrieval output
        formatted_results = []
        for result in results:
            formatted_results.append({
                'chunk': result['chunk'],
                'graph_score': 0.0,
                'embedding_score': result['score'],
                'combined_score': result['score'],
                'retrieval_method': 'vector_only'
            })

        return formatted_results

    def _enrich_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich results with full chunk data from index.

        Args:
            results: Results from reranking

        Returns:
            Enriched results with full chunk information
        """
        enriched = []

        for result in results:
            chunk_id = result['chunk_id']

            # Get full chunk data from index
            chunk = self.index_manager.get_chunk_by_id(chunk_id)

            if chunk:
                enriched.append({
                    'chunk': chunk,
                    'graph_score': result.get('graph_score', 0.0),
                    'embedding_score': result.get('embedding_score', 0.0),
                    'combined_score': result.get('combined_score', 0.0),
                    'ctv_matches': result.get('ctv_matches', []),
                    'retrieval_method': 'u_retrieval'
                })
            else:
                logger.warning(f"Chunk {chunk_id} not found in index")

        return enriched

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        max_candidates: int = 100
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query
            max_candidates: Max candidates from graph phase

        Returns:
            List of result lists (one per query)
        """
        all_results = []

        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            results = self.retrieve(query, top_k, max_candidates)
            all_results.append(results)

        return all_results


def main():
    parser = argparse.ArgumentParser(description="U-Retrieval pipeline")
    parser.add_argument(
        "--question-file",
        type=str,
        help="JSONL file with questions (each line: {\"id\": ..., \"question\": ...})"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query string (alternative to question-file)"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="index",
        help="Index directory"
    )
    parser.add_argument(
        "--graph",
        type=str,
        help="NetworkX graph pickle file"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        help="Neo4j URI (alternative to --graph)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL file for retrieval results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per query"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Max candidates from graph phase"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Graph score weight"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Embedding score weight"
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to process (for testing)"
    )

    args = parser.parse_args()

    # Import here to avoid circular dependencies
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from graph.query_graph import GraphQuerier
    from index.index_utils import load_index
    from retriever.top_down import TopDownRetriever
    from retriever.bottom_up import AdaptiveReranker

    # Load components
    logger.info("Loading retrieval components...")

    # Graph querier
    graph_querier = GraphQuerier(
        graph_path=Path(args.graph) if args.graph else None,
        neo4j_uri=args.neo4j_uri or os.getenv('NEO4J_URI')
    )

    # Index manager
    index_manager = load_index(args.index_dir)

    # Top-down retriever
    top_down = TopDownRetriever(graph_querier=graph_querier)
    if args.icd or args.mesh:
        top_down.load_ctv_mappings(
            icd_file=Path(args.icd) if args.icd else None,
            mesh_file=Path(args.mesh) if args.mesh else None
        )

    # Bottom-up reranker
    bottom_up = AdaptiveReranker(
        index_manager,
        base_alpha=args.alpha,
        base_beta=args.beta,
        adaptive=True
    )

    # U-Retriever
    retriever = URetriever(
        graph_querier=graph_querier,
        index_manager=index_manager,
        top_down_retriever=top_down,
        bottom_up_reranker=bottom_up
    )

    # Process queries
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as outf:
        if args.question_file:
            # Batch processing from file
            logger.info(f"Processing questions from {args.question_file}")
            if args.limit:
                logger.info(f"Limiting to {args.limit} questions")

            processed_count = 0
            with open(args.question_file, 'r') as inf:
                for line in inf:
                    # Check limit
                    if args.limit and processed_count >= args.limit:
                        logger.info(f"Reached limit of {args.limit} questions")
                        break

                    question_data = json.loads(line)
                    question_id = question_data.get('id', 'unknown')
                    question_text = question_data.get('question', '')

                    logger.info(f"Processing question {question_id} ({processed_count + 1}/{args.limit if args.limit else '?'})")

                    results = retriever.retrieve(
                        question_text,
                        top_k=args.top_k,
                        max_candidates=args.max_candidates
                    )

                    processed_count += 1

                    # Write results
                    output_data = {
                        'question_id': question_id,
                        'question': question_text,
                        'results': [
                            {
                                'chunk_id': r['chunk']['id'],
                                'text': r['chunk']['text'],
                                'source': r['chunk']['source'],
                                'source_id': r['chunk']['source_id'],
                                'graph_score': r['graph_score'],
                                'embedding_score': r['embedding_score'],
                                'combined_score': r['combined_score'],
                                'ctv_codes': r['chunk'].get('ctv_codes', [])
                            }
                            for r in results
                        ]
                    }

                    outf.write(json.dumps(output_data) + '\n')

        elif args.query:
            # Single query
            logger.info(f"Processing single query: {args.query}")

            results = retriever.retrieve(
                args.query,
                top_k=args.top_k,
                max_candidates=args.max_candidates
            )

            output_data = {
                'question_id': 'single_query',
                'question': args.query,
                'results': [
                    {
                        'chunk_id': r['chunk']['id'],
                        'text': r['chunk']['text'],
                        'source': r['chunk']['source'],
                        'source_id': r['chunk']['source_id'],
                        'graph_score': r['graph_score'],
                        'embedding_score': r['embedding_score'],
                        'combined_score': r['combined_score'],
                        'ctv_codes': r['chunk'].get('ctv_codes', [])
                    }
                    for r in results
                ]
            }

            outf.write(json.dumps(output_data, indent=2))
        else:
            logger.error("Must provide either --question-file or --query")
            return

    logger.info(f"Results written to {output_path}")

    # Cleanup
    graph_querier.close()


if __name__ == "__main__":
    main()
