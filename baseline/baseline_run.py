#!/usr/bin/env python3
"""
Baseline LLM evaluation without RAG.

Run Med-HALT questions through LLM without providing evidence
to establish baseline hallucination rates.
"""

import argparse
import logging
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from baseline_prompts import (
    build_baseline_prompt,
    build_medhalt_baseline_prompt
)

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_selected_option(response: str) -> Optional[str]:
    """
    Extract the selected option letter (A, B, C, or D) from a response.

    Args:
        response: Model response text

    Returns:
        Selected letter (A, B, C, or D) or None if not found
    """
    # First, look for a standalone letter at the beginning
    match = re.match(r'^([A-D])\b', response.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for "Answer: A" or "The answer is A" patterns
    patterns = [
        r'(?:answer|choice|option|select)(?:\s+is)?[\s:]+([A-D])\b',
        r'\b([A-D])\)?\s*[:-]?\s*(?:is|would be)?(?:\s+the)?(?:\s+correct)?(?:\s+answer)?',
        r'^([A-D])\)',  # Matches "A) ..."
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find any A, B, C, or D mentioned
    letters = re.findall(r'\b([A-D])\b', response, re.IGNORECASE)
    if letters:
        return letters[0].upper()

    return None


class BaselineRunner:
    """Run baseline LLM evaluation without RAG."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        mode: str = "zero-shot"
    ):
        """
        Initialize baseline runner.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            temperature: Sampling temperature
            max_tokens: Max response tokens
            mode: Prompt mode ("zero-shot", "few-shot", "medhalt")
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode

        # Set API key
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Initialize OpenAI client (new API)
        self.client = OpenAI(api_key=api_key)

        # Track if temperature is supported (to avoid repeated failures)
        self.temperature_supported = None  # None = unknown, True/False after first attempt

        logger.info(f"Initialized with model: {model}, mode: {mode}")

    def run_single_question(
        self,
        question: str,
        question_id: str = "unknown",
        options: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Generate response for a single question.

        Args:
            question: Question text
            question_id: Question identifier
            options: Multiple choice options (optional)

        Returns:
            Result dictionary
        """
        # Build prompt
        if self.mode == "medhalt":
            prompt = build_medhalt_baseline_prompt(question, options)
        else:
            prompt = build_baseline_prompt(question, mode=self.mode, options=options)

        # Call OpenAI API
        try:
            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['user']}
                ],
                "max_completion_tokens": self.max_tokens
            }

            # Only include temperature if it's supported and not default
            # Some models (like gpt-4o-mini) only support default temperature
            use_temperature = (
                self.temperature is not None and
                self.temperature != 1.0 and
                self.temperature_supported != False  # Only skip if we know it's not supported
            )

            if use_temperature:
                api_params["temperature"] = self.temperature

            try:
                response = self.client.chat.completions.create(**api_params)
                # If we get here with temperature, it's supported
                if use_temperature:
                    self.temperature_supported = True
            except Exception as api_error:
                # If temperature fails, retry without it
                if use_temperature and "temperature" in str(api_error):
                    logger.warning(f"Temperature {self.temperature} not supported for {self.model}, using default")
                    self.temperature_supported = False  # Remember for future calls
                    api_params.pop("temperature", None)
                    response = self.client.chat.completions.create(**api_params)
                else:
                    raise

            # Extract response
            response_text = response.choices[0].message.content.strip()

            result = {
                'id': question_id,
                'question': question,
                'response': response_text,
                'model': self.model,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat(),
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

            # If options provided, extract the selected option letter
            if options:
                selected = extract_selected_option(response_text)
                result['selected_option'] = selected

                # Convert letter to index (A->0, B->1, C->2, D->3)
                if selected:
                    result['selected_index'] = ord(selected) - ord('A')

            return result

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            return {
                'id': question_id,
                'question': question,
                'response': f"ERROR: {str(e)}",
                'model': self.model,
                'mode': self.mode,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run_batch(
        self,
        questions: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run baseline evaluation on multiple questions.

        Args:
            questions: List of question dicts with 'id', 'question', and optional 'options'
            show_progress: Show progress bar

        Returns:
            List of result dictionaries
        """
        results = []

        iterator = tqdm(questions, desc="Running baseline") if show_progress else questions

        for q in iterator:
            question_id = q.get('id', 'unknown')
            question_text = q.get('question', '')
            options = q.get('options')

            result = self.run_single_question(question_text, question_id, options)
            results.append(result)

        return results


def load_medhalt_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load Med-HALT dataset.

    Args:
        dataset_path: Path to Med-HALT JSONL file or directory

    Returns:
        List of questions with options
    """
    questions = []

    if dataset_path.is_file():
        # Single JSONL file
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Parse options if they exist
                options_str = data.get('options', '{}')
                if isinstance(options_str, str):
                    try:
                        import ast
                        options = ast.literal_eval(options_str)
                    except:
                        options = None
                else:
                    options = options_str

                questions.append({
                    'id': data.get('id', data.get('question_id', 'unknown')),
                    'question': data.get('question', data.get('text', '')),
                    'options': options
                })

    elif dataset_path.is_dir():
        # Directory with splits (e.g., train.jsonl, test.jsonl)
        for jsonl_file in dataset_path.glob("*.jsonl"):
            logger.info(f"Loading {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data = json.loads(line)

                    # Parse options if they exist
                    options_str = data.get('options', '{}')
                    if isinstance(options_str, str):
                        try:
                            import ast
                            options = ast.literal_eval(options_str)
                        except:
                            options = None
                    else:
                        options = options_str

                    questions.append({
                        'id': data.get('id', data.get('question_id', 'unknown')),
                        'question': data.get('question', data.get('text', '')),
                        'options': options
                    })
    else:
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    logger.info(f"Loaded {len(questions)} questions")
    return questions


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """
    Save baseline results to JSONL.

    Args:
        results: Result dictionaries
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline LLM evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Med-HALT dataset file or directory"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL file for results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero-shot", "few-shot", "medhalt"],
        default="zero-shot",
        help="Prompting mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum response tokens"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions (for testing)"
    )

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    questions = load_medhalt_dataset(Path(args.dataset))

    if not questions:
        logger.error("No questions found in dataset")
        return

    # Limit if specified
    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"Limited to {len(questions)} questions")

    # Initialize runner
    runner = BaselineRunner(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mode=args.mode
    )

    # Run evaluation
    logger.info(f"Running baseline evaluation on {len(questions)} questions...")
    results = runner.run_batch(questions, show_progress=True)

    # Save results
    save_results(results, Path(args.out))

    # Print summary
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    successful = total - errors

    avg_tokens = sum(
        r.get('usage', {}).get('total_tokens', 0)
        for r in results if 'usage' in r
    ) / max(successful, 1)

    logger.info("\n" + "="*60)
    logger.info("BASELINE EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Model:              {args.model}")
    logger.info(f"Mode:               {args.mode}")
    logger.info(f"Total questions:    {total}")
    logger.info(f"Successful:         {successful}")
    logger.info(f"Errors:             {errors}")
    logger.info(f"Avg tokens/question: {avg_tokens:.1f}")
    logger.info(f"Output file:        {args.out}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
