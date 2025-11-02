#!/usr/bin/env python3
"""
Abstention detection and quality scoring for RAG responses.

Implements:
1. Retrieval-based abstention (low retrieval scores)
2. Response-based abstention (LLM self-check)
3. Confidence estimation
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AbstentionDetector:
    """Detect when model should abstain from answering."""

    def __init__(
        self,
        retrieval_threshold: float = 0.3,
        response_threshold: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize abstention detector.

        Args:
            retrieval_threshold: Min score to proceed with generation
            response_threshold: Min confidence to accept response
            embedding_model: Model for semantic similarity
        """
        self.retrieval_threshold = retrieval_threshold
        self.response_threshold = response_threshold

        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

    def should_abstain_retrieval(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if retrieval quality is too low.

        Args:
            retrieval_results: Results from retrieval phase

        Returns:
            Dict with 'abstain' (bool) and 'reason' (str)
        """
        if not retrieval_results:
            return {
                'abstain': True,
                'reason': 'No relevant evidence found',
                'max_score': 0.0
            }

        # Get maximum retrieval score
        max_score = max(
            r.get('combined_score', r.get('score', 0))
            for r in retrieval_results
        )

        if max_score < self.retrieval_threshold:
            return {
                'abstain': True,
                'reason': f'Low retrieval confidence (max score: {max_score:.3f} < {self.retrieval_threshold})',
                'max_score': max_score
            }

        return {
            'abstain': False,
            'reason': None,
            'max_score': max_score
        }

    def should_abstain_response(
        self,
        question: str,
        response: str,
        evidence_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if response quality is suspect.

        Uses semantic similarity between response and evidence.

        Args:
            question: Original question
            response: Generated response
            evidence_chunks: Retrieved evidence

        Returns:
            Dict with 'abstain' (bool), 'reason' (str), 'confidence' (float)
        """
        # Check if response explicitly abstains
        if self._is_explicit_abstention(response):
            return {
                'abstain': True,
                'reason': 'Explicit abstention in response',
                'confidence': 0.0
            }

        # Compute semantic similarity to evidence
        confidence = self._compute_response_confidence(
            response,
            evidence_chunks
        )

        if confidence < self.response_threshold:
            return {
                'abstain': True,
                'reason': f'Low response confidence (score: {confidence:.3f} < {self.response_threshold})',
                'confidence': confidence
            }

        return {
            'abstain': False,
            'reason': None,
            'confidence': confidence
        }

    def _is_explicit_abstention(self, response: str) -> bool:
        """Check for explicit abstention keywords."""
        response_lower = response.lower()

        abstention_keywords = [
            'abstain',
            'cannot answer',
            'insufficient information',
            'insufficient evidence',
            'not enough information',
            'unable to answer',
            'cannot provide'
        ]

        return any(keyword in response_lower for keyword in abstention_keywords)

    def _compute_response_confidence(
        self,
        response: str,
        evidence_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Compute confidence score based on semantic similarity.

        Args:
            response: Generated response
            evidence_chunks: Evidence chunks

        Returns:
            Confidence score (0-1)
        """
        if not evidence_chunks:
            return 0.0

        # Encode response
        response_embedding = self.encoder.encode([response])

        # Encode evidence chunks
        evidence_texts = [chunk.get('text', '') for chunk in evidence_chunks]
        evidence_embeddings = self.encoder.encode(evidence_texts)

        # Compute similarities
        similarities = cosine_similarity(response_embedding, evidence_embeddings)[0]

        # Confidence = max similarity to evidence
        # (response should be semantically similar to at least one evidence chunk)
        confidence = float(np.max(similarities))

        return confidence

    def evaluate(
        self,
        question: str,
        response: str,
        retrieval_results: List[Dict[str, Any]],
        evidence_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Full abstention evaluation.

        Args:
            question: Question text
            response: Generated response
            retrieval_results: Retrieval results
            evidence_chunks: Evidence chunks used

        Returns:
            Evaluation dictionary
        """
        # Check retrieval quality
        retrieval_check = self.should_abstain_retrieval(retrieval_results)

        # Check response quality (if retrieval passed)
        if not retrieval_check['abstain']:
            response_check = self.should_abstain_response(
                question,
                response,
                evidence_chunks
            )
        else:
            response_check = {'abstain': False, 'reason': None, 'confidence': 0.0}

        # Overall decision
        should_abstain = retrieval_check['abstain'] or response_check['abstain']

        reasons = []
        if retrieval_check['abstain']:
            reasons.append(retrieval_check['reason'])
        if response_check['abstain']:
            reasons.append(response_check['reason'])

        return {
            'should_abstain': should_abstain,
            'reasons': reasons,
            'retrieval_score': retrieval_check['max_score'],
            'response_confidence': response_check.get('confidence', 0.0),
            'retrieval_check': retrieval_check,
            'response_check': response_check
        }


class ConfidenceEstimator:
    """Estimate confidence in generated responses."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize confidence estimator.

        Args:
            embedding_model: Sentence transformer model
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

    def estimate_confidence(
        self,
        question: str,
        response: str,
        evidence_chunks: List[Dict[str, Any]],
        retrieval_scores: List[float]
    ) -> Dict[str, float]:
        """
        Estimate multiple confidence signals.

        Args:
            question: Question text
            response: Generated response
            evidence_chunks: Evidence chunks
            retrieval_scores: Retrieval scores for evidence

        Returns:
            Dictionary of confidence metrics
        """
        # 1. Retrieval quality
        retrieval_confidence = np.mean(retrieval_scores) if retrieval_scores else 0.0

        # 2. Response-evidence alignment
        alignment_confidence = self._response_alignment(response, evidence_chunks)

        # 3. Response-question relevance
        relevance_confidence = self._response_relevance(question, response)

        # 4. Response length (heuristic: very short responses might be uncertain)
        length_score = min(len(response.split()) / 50.0, 1.0)  # Normalize to ~50 words

        # Overall confidence (weighted average)
        overall_confidence = (
            0.3 * retrieval_confidence +
            0.4 * alignment_confidence +
            0.2 * relevance_confidence +
            0.1 * length_score
        )

        return {
            'overall': overall_confidence,
            'retrieval': retrieval_confidence,
            'alignment': alignment_confidence,
            'relevance': relevance_confidence,
            'length_score': length_score
        }

    def _response_alignment(
        self,
        response: str,
        evidence_chunks: List[Dict[str, Any]]
    ) -> float:
        """Measure response-evidence semantic alignment."""
        if not evidence_chunks:
            return 0.0

        response_emb = self.encoder.encode([response])
        evidence_texts = [c.get('text', '') for c in evidence_chunks]
        evidence_embs = self.encoder.encode(evidence_texts)

        similarities = cosine_similarity(response_emb, evidence_embs)[0]
        return float(np.mean(similarities))

    def _response_relevance(self, question: str, response: str) -> float:
        """Measure response-question relevance."""
        question_emb = self.encoder.encode([question])
        response_emb = self.encoder.encode([response])

        similarity = cosine_similarity(question_emb, response_emb)[0][0]
        return float(similarity)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Abstention detection test")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--response", type=str, required=True)
    parser.add_argument("--retrieval-threshold", type=float, default=0.3)
    parser.add_argument("--response-threshold", type=float, default=0.5)

    args = parser.parse_args()

    # Mock evidence
    mock_evidence = [
        {
            'text': 'Diabetes is characterized by elevated blood glucose levels.',
            'source_id': 'PMID:12345',
            'source': 'pubmed'
        },
        {
            'text': 'Type 2 diabetes is the most common form of diabetes.',
            'source_id': 'PMID:67890',
            'source': 'pubmed'
        }
    ]

    mock_retrieval_results = [
        {'combined_score': 0.8},
        {'combined_score': 0.6}
    ]

    # Initialize detector
    detector = AbstentionDetector(
        retrieval_threshold=args.retrieval_threshold,
        response_threshold=args.response_threshold
    )

    # Evaluate
    evaluation = detector.evaluate(
        args.question,
        args.response,
        mock_retrieval_results,
        mock_evidence
    )

    print("\n" + "="*60)
    print("ABSTENTION EVALUATION")
    print("="*60)
    print(f"Should abstain: {evaluation['should_abstain']}")
    print(f"Retrieval score: {evaluation['retrieval_score']:.3f}")
    print(f"Response confidence: {evaluation['response_confidence']:.3f}")

    if evaluation['reasons']:
        print(f"\nReasons:")
        for reason in evaluation['reasons']:
            print(f"  - {reason}")

    # Confidence estimation
    estimator = ConfidenceEstimator()
    confidence = estimator.estimate_confidence(
        args.question,
        args.response,
        mock_evidence,
        [0.8, 0.6]
    )

    print("\n" + "="*60)
    print("CONFIDENCE METRICS")
    print("="*60)
    for metric, score in confidence.items():
        print(f"{metric:15s}: {score:.3f}")
