#!/usr/bin/env python3
"""
Med-HALT scoring with penalized accuracy.

Implements:
- Penalized scoring: +1 correct, -0.25 incorrect, 0 abstain
- RHT / MHT breakdown
- Statistical significance testing
"""

import argparse
import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedHALTScorer:
    """Score model outputs using Med-HALT methodology."""

    def __init__(
        self,
        correct_score: float = 1.0,
        incorrect_penalty: float = -0.25,
        abstain_score: float = 0.0
    ):
        """
        Initialize scorer with penalty values.

        Args:
            correct_score: Points for correct answer
            incorrect_penalty: Penalty for incorrect answer
            abstain_score: Score for abstention
        """
        self.correct_score = correct_score
        self.incorrect_penalty = incorrect_penalty
        self.abstain_score = abstain_score

        logger.info(
            f"Scorer initialized: +{correct_score} correct, "
            f"{incorrect_penalty} incorrect, {abstain_score} abstain"
        )

    def score_response(
        self,
        response: str,
        ground_truth: str,
        question_type: str = None,
        selected_index: int = None,
        correct_index: int = None
    ) -> Dict[str, Any]:
        """
        Score a single response.

        Args:
            response: Model response
            ground_truth: Correct answer
            question_type: "RHT" or "MHT"
            selected_index: Index of selected option (for MC questions)
            correct_index: Index of correct option (for MC questions)

        Returns:
            Scoring result dictionary
        """
        # Check if abstention
        is_abstain = self._is_abstention(response)

        if is_abstain:
            return {
                'score': self.abstain_score,
                'correct': None,
                'abstained': True,
                'question_type': question_type
            }

        # Evaluate correctness
        is_correct = self._evaluate_correctness(
            response,
            ground_truth,
            selected_index,
            correct_index
        )

        score = self.correct_score if is_correct else self.incorrect_penalty

        return {
            'score': score,
            'correct': is_correct,
            'abstained': False,
            'question_type': question_type
        }

    def _is_abstention(self, response: str) -> bool:
        """Check if response is an abstention."""
        response_lower = response.lower().strip()

        abstention_keywords = [
            'abstain',
            "i don't know",
            "i'm not certain",
            "insufficient information",
            "insufficient evidence",
            "cannot answer",
            "unable to answer"
        ]

        return any(keyword in response_lower for keyword in abstention_keywords)

    def _evaluate_correctness(
        self,
        response: str,
        ground_truth: str,
        selected_index: int = None,
        correct_index: int = None
    ) -> bool:
        """
        Evaluate if response is correct.

        For multiple choice: uses exact index matching
        For free text: uses fuzzy term matching (less reliable)

        Args:
            response: Model response
            ground_truth: Ground truth answer
            selected_index: Index of selected option (0-3) for MC questions
            correct_index: Index of correct option (0-3) for MC questions

        Returns:
            True if correct
        """
        # If we have index information (multiple choice), use exact matching
        if selected_index is not None and correct_index is not None:
            return selected_index == correct_index

        # Fallback to text matching (less reliable)
        if not ground_truth:
            return False

        response_lower = response.lower()
        truth_lower = ground_truth.lower()

        # Extract key terms
        truth_terms = set(truth_lower.split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        truth_terms = truth_terms - stop_words

        if not truth_terms:
            return False

        # Check overlap (heuristic)
        matches = sum(1 for term in truth_terms if term in response_lower)

        # Require at least 50% term overlap as a simple heuristic
        threshold = len(truth_terms) * 0.5

        return matches >= threshold

    def score_dataset(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Score entire dataset.

        Args:
            predictions: List of prediction dicts with 'id' and 'response'
            ground_truth: List of ground truth dicts with 'id' and 'answer'

        Returns:
            Comprehensive scoring results
        """
        # Create lookup for ground truth
        gt_lookup = {item['id']: item for item in ground_truth}

        # Score each prediction
        results = []
        for pred in predictions:
            pred_id = pred['id']
            response = pred.get('response', '')

            # Find matching ground truth
            if pred_id not in gt_lookup:
                logger.warning(f"No ground truth found for ID: {pred_id}")
                continue

            gt = gt_lookup[pred_id]
            answer = gt.get('answer', '')
            question_type = gt.get('type')

            # Get indices for multiple choice questions
            selected_index = pred.get('selected_index')
            correct_index = gt.get('correct_index')

            # Score
            score_result = self.score_response(
                response,
                answer,
                question_type,
                selected_index,
                correct_index
            )

            results.append({
                'id': pred_id,
                **score_result,
                'response': response,
                'ground_truth': answer,
                'selected_index': selected_index,
                'correct_index': correct_index
            })

        # Compute aggregate metrics
        metrics = self._compute_metrics(results)

        return {
            'individual_scores': results,
            'metrics': metrics
        }

    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        total = len(results)

        if total == 0:
            logger.error("No results to compute metrics. This likely means no predictions matched ground truth IDs.")
            return {
                'penalized_accuracy': 0.0,
                'total_questions': 0,
                'correct': 0,
                'incorrect': 0,
                'abstained': 0,
                'abstention_rate': 0.0,
                'accuracy_on_answered': 0.0,
                'error': 'No matched results'
            }

        # Overall scores
        total_score = sum(r['score'] for r in results)
        penalized_accuracy = total_score / total

        # Breakdown by correctness
        correct = sum(1 for r in results if r.get('correct') == True)
        incorrect = sum(1 for r in results if r.get('correct') == False)
        abstained = sum(1 for r in results if r.get('abstained') == True)

        # Breakdown by question type
        rht_results = [r for r in results if r.get('question_type') == 'RHT']
        mht_results = [r for r in results if r.get('question_type') == 'MHT']

        rht_score = sum(r['score'] for r in rht_results) / len(rht_results) if rht_results else 0
        mht_score = sum(r['score'] for r in mht_results) / len(mht_results) if mht_results else 0

        # Abstention metrics
        abstention_rate = abstained / total
        precision_on_answered = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0

        return {
            'penalized_accuracy': penalized_accuracy,
            'total_questions': total,
            'correct': correct,
            'incorrect': incorrect,
            'abstained': abstained,
            'accuracy_on_answered': precision_on_answered,
            'abstention_rate': abstention_rate,
            'rht': {
                'count': len(rht_results),
                'penalized_accuracy': rht_score
            },
            'mht': {
                'count': len(mht_results),
                'penalized_accuracy': mht_score
            }
        }

    def compare_systems(
        self,
        baseline_results: Dict[str, Any],
        rag_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline vs RAG systems.

        Args:
            baseline_results: Baseline scoring results
            rag_results: RAG scoring results

        Returns:
            Comparison statistics
        """
        baseline_metrics = baseline_results['metrics']
        rag_metrics = rag_results['metrics']

        # Compute improvements
        improvement = {
            'penalized_accuracy': rag_metrics['penalized_accuracy'] - baseline_metrics['penalized_accuracy'],
            'abstention_rate': rag_metrics['abstention_rate'] - baseline_metrics['abstention_rate'],
            'rht_improvement': rag_metrics['rht']['penalized_accuracy'] - baseline_metrics['rht']['penalized_accuracy'],
            'mht_improvement': rag_metrics['mht']['penalized_accuracy'] - baseline_metrics['mht']['penalized_accuracy']
        }

        # Statistical significance (McNemar's test for paired binary outcomes)
        baseline_correct = [
            r.get('correct', False)
            for r in baseline_results['individual_scores']
        ]
        rag_correct = [
            r.get('correct', False)
            for r in rag_results['individual_scores']
        ]

        # Align by ID
        baseline_by_id = {r['id']: r.get('correct', False) for r in baseline_results['individual_scores']}
        rag_by_id = {r['id']: r.get('correct', False) for r in rag_results['individual_scores']}

        common_ids = set(baseline_by_id.keys()) & set(rag_by_id.keys())

        if len(common_ids) > 0:
            baseline_aligned = [baseline_by_id[id] for id in common_ids]
            rag_aligned = [rag_by_id[id] for id in common_ids]

            # McNemar's test
            # Build contingency table
            both_correct = sum(1 for i in range(len(common_ids)) if baseline_aligned[i] and rag_aligned[i])
            both_incorrect = sum(1 for i in range(len(common_ids)) if not baseline_aligned[i] and not rag_aligned[i])
            baseline_only = sum(1 for i in range(len(common_ids)) if baseline_aligned[i] and not rag_aligned[i])
            rag_only = sum(1 for i in range(len(common_ids)) if not baseline_aligned[i] and rag_aligned[i])

            # McNemar test statistic
            if baseline_only + rag_only > 0:
                mcnemar_stat = (abs(baseline_only - rag_only) - 1) ** 2 / (baseline_only + rag_only)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            else:
                p_value = 1.0

            improvement['statistical_significance'] = {
                'mcnemar_p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'contingency': {
                    'both_correct': both_correct,
                    'both_incorrect': both_incorrect,
                    'baseline_only_correct': baseline_only,
                    'rag_only_correct': rag_only
                }
            }

        return {
            'baseline': baseline_metrics,
            'rag': rag_metrics,
            'improvement': improvement
        }


def load_predictions(file_path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions.append({
                'id': data.get('id', data.get('question_id')),
                'response': data.get('response', ''),
                'question': data.get('question', '')
            })

    return predictions


def load_ground_truth(file_path: Path) -> List[Dict[str, Any]]:
    """Load ground truth from JSONL file."""
    ground_truth = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Parse options if they exist (for multiple choice)
            options_str = data.get('options', '{}')
            if isinstance(options_str, str):
                try:
                    import ast
                    options = ast.literal_eval(options_str)
                except:
                    options = {}
            else:
                options = options_str

            ground_truth.append({
                'id': data.get('id', data.get('question_id')),
                'answer': data.get('correct_answer', data.get('answer', '')),
                'correct_index': data.get('correct_index'),
                'options': options,
                'type': data.get('type'),
                'category': data.get('category')
            })

    return ground_truth


def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_metrics(metrics: Dict[str, Any], output_file: Path):
    """Save metrics to CSV and JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types
    metrics = _convert_numpy_types(metrics)

    # Save as JSON
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save summary as CSV
    csv_file = output_file
    if 'baseline' in metrics:
        # Comparison format
        df_data = {
            'Metric': [],
            'Baseline': [],
            'RAG': [],
            'Improvement': []
        }

        for key in ['penalized_accuracy', 'abstention_rate']:
            df_data['Metric'].append(key)
            df_data['Baseline'].append(metrics['baseline'].get(key, 0))
            df_data['RAG'].append(metrics['rag'].get(key, 0))
            df_data['Improvement'].append(metrics['improvement'].get(key, 0))

        df = pd.DataFrame(df_data)
    else:
        # Single system format
        df_data = {'Metric': [], 'Value': []}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                df_data['Metric'].append(key)
                df_data['Value'].append(value)

        df = pd.DataFrame(df_data)

    df.to_csv(csv_file, index=False)

    logger.info(f"Metrics saved to {json_file} and {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Med-HALT scoring")
    parser.add_argument("--pred", type=str, required=True, help="Predictions JSONL file")
    parser.add_argument("--gold", type=str, required=True, help="Ground truth JSONL file")
    parser.add_argument("--out", type=str, required=True, help="Output file (CSV/JSON)")
    parser.add_argument("--baseline-pred", type=str, help="Baseline predictions for comparison")

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading predictions from {args.pred}")
    predictions = load_predictions(Path(args.pred))

    logger.info(f"Loading ground truth from {args.gold}")
    ground_truth = load_ground_truth(Path(args.gold))

    # Initialize scorer
    scorer = MedHALTScorer()

    # Score predictions
    logger.info("Scoring predictions...")
    results = scorer.score_dataset(predictions, ground_truth)

    # If baseline provided, compute comparison
    if args.baseline_pred:
        logger.info(f"Loading baseline predictions from {args.baseline_pred}")
        baseline_predictions = load_predictions(Path(args.baseline_pred))

        logger.info("Scoring baseline...")
        baseline_results = scorer.score_dataset(baseline_predictions, ground_truth)

        logger.info("Computing comparison...")
        metrics = scorer.compare_systems(baseline_results, results)
    else:
        metrics = results['metrics']

    # Save metrics
    save_metrics(metrics, Path(args.out))

    # Print summary
    print("\n" + "="*60)
    print("SCORING RESULTS")
    print("="*60)

    if 'baseline' in metrics:
        print(f"\nBaseline Penalized Accuracy:  {metrics['baseline']['penalized_accuracy']:.4f}")
        print(f"RAG Penalized Accuracy:       {metrics['rag']['penalized_accuracy']:.4f}")
        print(f"Improvement:                  {metrics['improvement']['penalized_accuracy']:+.4f}")

        if 'statistical_significance' in metrics['improvement']:
            sig = metrics['improvement']['statistical_significance']
            print(f"\nStatistical Significance:")
            print(f"  McNemar p-value:  {sig['mcnemar_p_value']:.4f}")
            print(f"  Significant:      {sig['significant_at_0.05']}")
    else:
        print(f"\nPenalized Accuracy:  {metrics['penalized_accuracy']:.4f}")
        print(f"Abstention Rate:     {metrics['abstention_rate']:.2%}")
        print(f"Accuracy (answered): {metrics['accuracy_on_answered']:.2%}")

    print("="*60)


if __name__ == "__main__":
    main()
