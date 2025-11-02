#!/usr/bin/env python3
"""
RAG response generation with citation enforcement and abstention.

Generates answers using:
1. Retrieved evidence chunks
2. Citation-enforcing prompts
3. Optional abstention detection
"""

import argparse
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompt_templates import (
    build_prompt,
    validate_response,
    is_abstention,
    extract_citations
)

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGGenerator:
    """Generate responses using retrieved evidence."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        template_type: str = "strict",
        enable_validation: bool = True
    ):
        """
        Initialize RAG generator.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or from environment)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            template_type: Prompt template ("default" or "strict")
            enable_validation: Validate responses for citations
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.template_type = template_type
        self.enable_validation = enable_validation

        # Set API key
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Initialize OpenAI client (new API)
        self.client = OpenAI(api_key=api_key)

        # Track if temperature is supported (to avoid repeated failures)
        self.temperature_supported = None  # None = unknown, True/False after first attempt

        logger.info(f"Initialized with model: {model}")

    def generate(
        self,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        include_few_shot: bool = False,
        question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response for a question with evidence.

        Args:
            question: Question text
            evidence_chunks: Retrieved evidence chunks
            include_few_shot: Include few-shot examples in prompt
            question_id: Optional question ID for tracking

        Returns:
            Generation result dictionary
        """
        # Build prompt
        prompt = build_prompt(
            question,
            evidence_chunks,
            template=self.template_type,
            include_few_shot=include_few_shot
        )

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

            # Extract response text
            response_text = response.choices[0].message.content.strip()

            # Validate response
            validation = None
            if self.enable_validation:
                validation = validate_response(response_text, len(evidence_chunks))
                # Ensure validation is always a dict, never a plain bool
                if not isinstance(validation, dict):
                    validation = {'valid': bool(validation), 'error': 'Invalid validation result type'}

            # Build result
            result = {
                'question_id': question_id,
                'question': question,
                'response': response_text,
                'model': self.model,
                'abstained': is_abstention(response_text),
                'num_evidence_chunks': len(evidence_chunks),
                'evidence_chunks': evidence_chunks,
                'validation': validation,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

            # Add citations if not abstained
            if not result['abstained']:
                result['citations'] = extract_citations(response_text)

            return result

        except Exception as e:
            import traceback
            logger.error(f"Error generating response: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'question_id': question_id,
                'question': question,
                'response': f"ERROR: {str(e)}",
                'model': self.model,
                'abstained': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def batch_generate(
        self,
        questions_with_evidence: List[Dict[str, Any]],
        include_few_shot: bool = False,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple questions.

        Args:
            questions_with_evidence: List of dicts with 'question' and 'evidence'
            include_few_shot: Include few-shot examples
            show_progress: Show progress bar

        Returns:
            List of generation results
        """
        results = []

        iterator = tqdm(questions_with_evidence) if show_progress else questions_with_evidence

        for item in iterator:
            question_id = item.get('question_id')
            question = item['question']
            evidence = item.get('evidence', item.get('results', []))

            result = self.generate(question, evidence, include_few_shot, question_id)
            results.append(result)

        return results


def load_retrieval_results(input_file: Path) -> List[Dict[str, Any]]:
    """
    Load retrieval results from JSONL file.

    Args:
        input_file: Path to retrieval results JSONL

    Returns:
        List of question + evidence dictionaries
    """
    questions_with_evidence = []

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Extract question and evidence chunks
            question = data.get('question', '')
            results = data.get('results', [])

            # Format evidence chunks
            evidence_chunks = [
                {
                    'text': r.get('text', ''),
                    'source_id': r.get('source_id', ''),
                    'source': r.get('source', ''),
                    'ctv_codes': r.get('ctv_codes', [])
                }
                for r in results
            ]

            questions_with_evidence.append({
                'question_id': data.get('question_id', 'unknown'),
                'question': question,
                'evidence': evidence_chunks
            })

    logger.info(f"Loaded {len(questions_with_evidence)} questions with evidence")
    return questions_with_evidence


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """
    Save generation results to JSONL file.

    Args:
        results: Generation results
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="RAG response generation")
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Input JSONL file with retrieval results"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL file for generated responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum response tokens"
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["default", "strict", "lenient"],
        default="strict",
        help="Prompt template type (strict=conservative, lenient=less abstention)"
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Include few-shot examples in prompts"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable response validation"
    )

    args = parser.parse_args()

    # Load retrieval results
    logger.info(f"Loading retrieval results from {args.candidates}")
    questions_with_evidence = load_retrieval_results(Path(args.candidates))

    if not questions_with_evidence:
        logger.error("No questions found in input file")
        return

    # Initialize generator
    generator = RAGGenerator(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        template_type=args.template,
        enable_validation=not args.no_validation
    )

    # Generate responses
    logger.info(f"Generating responses using {args.model}...")
    results = generator.batch_generate(
        questions_with_evidence,
        include_few_shot=args.few_shot,
        show_progress=True
    )

    # Save results
    save_results(results, Path(args.out))

    # Print summary statistics
    total = len(results)
    abstained = sum(1 for r in results if r.get('abstained', False))
    errors = sum(1 for r in results if 'error' in r)

    if args.no_validation:
        validated = 0
        invalid = 0
    else:
        validated = sum(
            1 for r in results
            if isinstance(r.get('validation'), dict) and r.get('validation', {}).get('valid', False)
        )
        invalid = total - abstained - errors - validated

    logger.info("\n" + "="*60)
    logger.info("GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total questions:    {total}")
    logger.info(f"Successful:         {total - errors}")
    logger.info(f"Abstentions:        {abstained} ({100*abstained/total:.1f}%)")
    logger.info(f"Validated:          {validated} ({100*validated/total:.1f}%)")
    logger.info(f"Invalid citations:  {invalid} ({100*invalid/total:.1f}%)")
    logger.info(f"Errors:             {errors}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
