#!/usr/bin/env python3
"""
Orchestrate complete experimental pipeline:
1. Baseline evaluation
2. RAG evaluation
3. Comparison and analysis
"""

import argparse
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Orchestrate full experimental pipeline."""

    def __init__(self, config_path: Path):
        """
        Initialize orchestrator with config.

        Args:
            config_path: Path to experiment configuration JSON
        """
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.results_dir = Path(self.config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_baseline(self) -> Path:
        """Run baseline evaluation."""
        logger.info("="*60)
        logger.info("PHASE 1: Baseline Evaluation")
        logger.info("="*60)

        baseline_config = self.config['baseline']
        dataset = baseline_config['dataset']
        model = baseline_config.get('model', 'gpt-3.5-turbo')
        provider = baseline_config.get('provider', 'openai')
        mode = baseline_config.get('mode', 'zero-shot')

        output_file = self.results_dir / f"baseline_{self.timestamp}.jsonl"

        cmd = [
            'python', 'baseline/baseline_run.py',
            '--dataset', dataset,
            '--model', model,
            '--provider', provider,
            '--mode', mode,
            '--out', str(output_file)
        ]

        if 'limit' in baseline_config:
            cmd.extend(['--limit', str(baseline_config['limit'])])

        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        logger.info(f"Baseline results saved to {output_file}")
        return output_file

    def run_rag_pipeline(self) -> Path:
        """Run complete RAG pipeline (retrieval + generation)."""
        logger.info("="*60)
        logger.info("PHASE 2: RAG Pipeline")
        logger.info("="*60)

        rag_config = self.config['rag']
        dataset = rag_config['dataset']

        # Step 1: Retrieval
        logger.info("Step 2.1: Retrieval")
        retrieval_output = self.results_dir / f"rag_retrieval_{self.timestamp}.jsonl"

        retrieval_cmd = [
            'python', 'retriever/retrieve.py',
            '--question-file', dataset,
            '--index-dir', rag_config.get('index_dir', 'index'),
            '--out', str(retrieval_output),
            '--top-k', str(rag_config.get('top_k', 10))
        ]

        if 'neo4j_uri' in rag_config:
            retrieval_cmd.extend(['--neo4j-uri', rag_config['neo4j_uri']])
        elif 'graph' in rag_config:
            retrieval_cmd.extend(['--graph', rag_config['graph']])

        logger.info(f"Running: {' '.join(retrieval_cmd)}")
        subprocess.run(retrieval_cmd, check=True)

        # Step 2: Generation
        logger.info("Step 2.2: Generation")
        generation_output = self.results_dir / f"rag_generation_{self.timestamp}.jsonl"

        generation_cmd = [
            'python', 'rag/generate.py',
            '--candidates', str(retrieval_output),
            '--model', rag_config.get('model', 'gpt-3.5-turbo'),
            '--provider', rag_config.get('provider', 'openai'),
            '--template', rag_config.get('template', 'strict'),
            '--out', str(generation_output)
        ]

        logger.info(f"Running: {' '.join(generation_cmd)}")
        subprocess.run(generation_cmd, check=True)

        logger.info(f"RAG results saved to {generation_output}")
        return generation_output

    def run_evaluation(
        self,
        baseline_results: Path,
        rag_results: Path
    ) -> Path:
        """Run comparative evaluation."""
        logger.info("="*60)
        logger.info("PHASE 3: Evaluation")
        logger.info("="*60)

        eval_config = self.config['evaluation']
        ground_truth = eval_config['ground_truth']

        # Score baseline
        logger.info("Scoring baseline...")
        baseline_scores = self.results_dir / f"baseline_scores_{self.timestamp}.csv"

        baseline_cmd = [
            'python', 'evaluation/scorer.py',
            '--pred', str(baseline_results),
            '--gold', ground_truth,
            '--out', str(baseline_scores)
        ]

        subprocess.run(baseline_cmd, check=True)

        # Score RAG with comparison
        logger.info("Scoring RAG with comparison...")
        comparison_scores = self.results_dir / f"comparison_{self.timestamp}.csv"

        rag_cmd = [
            'python', 'evaluation/scorer.py',
            '--pred', str(rag_results),
            '--gold', ground_truth,
            '--baseline-pred', str(baseline_results),
            '--out', str(comparison_scores)
        ]

        subprocess.run(rag_cmd, check=True)

        logger.info(f"Comparison results saved to {comparison_scores}")
        return comparison_scores

    def generate_report(self, comparison_file: Path):
        """Generate experiment report."""
        logger.info("="*60)
        logger.info("PHASE 4: Report Generation")
        logger.info("="*60)

        # Load comparison results
        with open(comparison_file.with_suffix('.json'), 'r') as f:
            results = json.load(f)

        report_file = self.results_dir / f"experiment_report_{self.timestamp}.md"

        report_content = f"""# Med-HALT RAG Experiment Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Configuration:** {self.config.get('name', 'Default')}

## Results Summary

### Overall Performance

| Metric | Baseline | RAG | Improvement |
|--------|----------|-----|-------------|
| Penalized Accuracy | {results['baseline']['penalized_accuracy']:.4f} | {results['rag']['penalized_accuracy']:.4f} | {results['improvement']['penalized_accuracy']:+.4f} |
| Abstention Rate | {results['baseline']['abstention_rate']:.2%} | {results['rag']['abstention_rate']:.2%} | {results['improvement']['abstention_rate']:+.2%} |

### Breakdown by Question Type

#### Reasoning Hallucination (RHT)
- Baseline: {results['baseline']['rht']['penalized_accuracy']:.4f}
- RAG: {results['rag']['rht']['penalized_accuracy']:.4f}
- Improvement: {results['improvement']['rht_improvement']:+.4f}

#### Memory Hallucination (MHT)
- Baseline: {results['baseline']['mht']['penalized_accuracy']:.4f}
- RAG: {results['rag']['mht']['penalized_accuracy']:.4f}
- Improvement: {results['improvement']['mht_improvement']:+.4f}

### Statistical Significance

"""
        if 'statistical_significance' in results['improvement']:
            sig = results['improvement']['statistical_significance']
            report_content += f"""- McNemar p-value: {sig['mcnemar_p_value']:.4f}
- Significant at α=0.05: {'Yes' if sig['significant_at_0.05'] else 'No'}

"""

        report_content += f"""
## Configuration Details

```json
{json.dumps(self.config, indent=2)}
```

## Files

- Baseline results: `{self.results_dir}/baseline_{self.timestamp}.jsonl`
- RAG retrieval: `{self.results_dir}/rag_retrieval_{self.timestamp}.jsonl`
- RAG generation: `{self.results_dir}/rag_generation_{self.timestamp}.jsonl`
- Comparison: `{self.results_dir}/comparison_{self.timestamp}.csv`

---
Generated with Med-HALT RAG PoC
"""

        with open(report_file, 'w') as f:
            f.write(report_content)

        logger.info(f"Report saved to {report_file}")
        return report_file

    def run_full_experiment(self):
        """Run complete experimental pipeline."""
        logger.info("Starting full experimental pipeline")
        logger.info(f"Timestamp: {self.timestamp}")

        try:
            # Phase 1: Baseline
            baseline_results = self.run_baseline()

            # Phase 2: RAG
            rag_results = self.run_rag_pipeline()

            # Phase 3: Evaluation
            comparison_results = self.run_evaluation(baseline_results, rag_results)

            # Phase 4: Report
            report = self.generate_report(comparison_results)

            logger.info("="*60)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Report: {report}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment failed: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run complete Med-HALT RAG experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.json",
        help="Experiment configuration file"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    orchestrator = ExperimentOrchestrator(config_path)
    orchestrator.run_full_experiment()


if __name__ == "__main__":
    main()
