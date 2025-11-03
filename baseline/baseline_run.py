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
from google import genai
from google.genai import types
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
        mode: str = "zero-shot",
        provider: str = "openai"
    ):
        """
        Initialize baseline runner.

        Args:
            model: Model name (e.g., gpt-3.5-turbo for OpenAI, gemini-pro for Gemini)
            api_key: API key
            temperature: Sampling temperature
            max_tokens: Max response tokens
            mode: Prompt mode ("zero-shot", "few-shot", "medhalt")
            provider: Model provider ("openai" or "gemini")
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode
        self.provider = provider.lower()

        # Initialize appropriate client based on provider
        if self.provider == "openai":
            # Set API key
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

            # Initialize OpenAI client
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "gemini":
            # Set API key
            api_key = api_key or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")

            # Initialize Gemini client with the new google.genai library
            self.client = genai.Client(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'gemini'.")

        # Track if temperature is supported (to avoid repeated failures)
        self.temperature_supported = None  # None = unknown, True/False after first attempt

        logger.info(f"Initialized with provider: {self.provider}, model: {model}, mode: {mode}")

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

        # Call appropriate API based on provider
        try:
            if self.provider == "openai":
                # Build API call parameters for OpenAI
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

                # Extract token usage
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }

            elif self.provider == "gemini":
                # Build prompt for Gemini (combine system and user prompts)
                full_prompt = f"{prompt['system']}\n\n{prompt['user']}"

                # Configure generation parameters using the new google.genai types
                config_params = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "safety_settings": [
                        types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='OFF'),
                        types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'),
                        types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF'),
                        types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'),
                    ]
                }

                generation_config = types.GenerateContentConfig(**config_params)

                try:
                    # Use the new google.genai API
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=full_prompt,
                        config=generation_config
                    )
                    self.temperature_supported = True
                except Exception as api_error:
                    logger.error(f"Gemini API error: {api_error}")
                    # If temperature fails, retry without it
                    if "temperature" in str(api_error):
                        logger.warning(f"Temperature {self.temperature} not supported for {self.model}, using default")
                        self.temperature_supported = False
                        config_params.pop("temperature", None)
                        generation_config = types.GenerateContentConfig(**config_params)
                        response = self.client.models.generate_content(
                            model=self.model,
                            contents=full_prompt,
                            config=generation_config
                        )
                    else:
                        raise

                # Extract response text with error handling
                try:
                    # Try to get text directly
                    if hasattr(response, 'text') and response.text:
                        response_text = response.text.strip()
                    else:
                        raise ValueError("response.text is None or empty")
                except Exception as text_error:
                    # Handle cases where response.text is not available
                    logger.warning(f"Failed to extract text from Gemini response: {text_error}")

                    response_text = None

                    # Check if response has candidates
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]

                        # Get finish_reason (it's an enum/string in new API, not int)
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = str(candidate.finish_reason)
                            logger.error(f"Gemini finish_reason: {finish_reason}")

                            # Check if it's MAX_TOKENS
                            if 'MAX_TOKENS' in finish_reason:
                                logger.error(f"Hit max tokens limit! Current limit: {self.max_tokens}")

                        # Try to extract partial text if available
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            if parts and len(parts) > 0 and hasattr(parts[0], 'text'):
                                response_text = parts[0].text.strip()
                                logger.info("Extracted partial text from response")

                    # If still no text, set error message
                    if not response_text:
                        finish_reason_str = str(candidate.finish_reason) if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason') else "unknown"
                        logger.error("No text found in response")
                        response_text = f"ERROR: No text in Gemini response (finish_reason: {finish_reason_str})"

                # Extract token usage (Gemini provides this differently)
                usage_info = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    'completion_tokens': response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    'total_tokens': response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                }

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            result = {
                'id': question_id,
                'question': question,
                'response': response_text,
                'model': self.model,
                'provider': self.provider,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat(),
                'usage': usage_info
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
        help="Model name (e.g., gpt-3.5-turbo for OpenAI, gemini-pro for Gemini)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="Model provider (openai or gemini)"
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
        provider=args.provider,
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
