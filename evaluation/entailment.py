#!/usr/bin/env python3
"""
Entailment checking for claim verification.

Note: Stub implementation. Full version would use:
- Fine-tuned NLI models (e.g., BioMed-RoBERTa-NLI)
- Med-HALT's entailment methodology
"""

import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntailmentChecker:
    """Check if evidence entails claims."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize entailment checker with embedding model."""
        logger.info(f"Loading model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

    def check_entailment(
        self,
        claim: str,
        evidence_chunks: List[str],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Check if evidence supports claim.

        Args:
            claim: Claim to verify
            evidence_chunks: List of evidence texts
            threshold: Similarity threshold for support

        Returns:
            Entailment result
        """
        if not evidence_chunks:
            return {
                'supported': False,
                'max_similarity': 0.0,
                'label': 'not_supported'
            }

        # Encode claim and evidence
        claim_emb = self.encoder.encode([claim])
        evidence_embs = self.encoder.encode(evidence_chunks)

        # Compute similarities
        similarities = cosine_similarity(claim_emb, evidence_embs)[0]
        max_sim = float(max(similarities))

        # Simple threshold-based decision
        supported = max_sim >= threshold

        return {
            'supported': supported,
            'max_similarity': max_sim,
            'label': 'supported' if supported else 'not_supported',
            'best_evidence_idx': int(similarities.argmax())
        }


if __name__ == "__main__":
    checker = EntailmentChecker()

    claim = "Metformin is used to treat type 2 diabetes."
    evidence = [
        "Metformin is a first-line medication for type 2 diabetes management.",
        "Insulin is used to treat type 1 diabetes.",
        "Exercise helps manage blood sugar levels."
    ]

    result = checker.check_entailment(claim, evidence)
    print(f"\nClaim: {claim}")
    print(f"Supported: {result['supported']}")
    print(f"Max similarity: {result['max_similarity']:.3f}")
    print(f"Best evidence: {evidence[result['best_evidence_idx']]}")
