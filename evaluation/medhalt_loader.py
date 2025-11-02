#!/usr/bin/env python3
"""
Load and process Med-HALT dataset.

Handles different Med-HALT formats and splits.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedHALTLoader:
    """Load and manage Med-HALT dataset."""

    def __init__(self, dataset_path: Optional[Path] = None, use_huggingface: bool = True):
        """
        Initialize Med-HALT loader.

        Args:
            dataset_path: Local path to Med-HALT data
            use_huggingface: Load from Hugging Face if True
        """
        self.dataset_path = dataset_path
        self.use_huggingface = use_huggingface
        self.dataset = None

    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load Med-HALT dataset.

        Args:
            split: Dataset split ("train", "validation", "test")

        Returns:
            List of question dictionaries
        """
        if self.use_huggingface:
            return self._load_from_huggingface(split)
        elif self.dataset_path:
            return self._load_from_local(split)
        else:
            raise ValueError("Must provide dataset_path or set use_huggingface=True")

    def _load_from_huggingface(self, split: str) -> List[Dict[str, Any]]:
        """Load from Hugging Face."""
        logger.info(f"Loading Med-HALT from Hugging Face (split: {split})")

        try:
            dataset = load_dataset("openlifescienceai/Med-HALT", split=split)
            self.dataset = dataset

            questions = []
            for item in dataset:
                questions.append({
                    'id': item.get('id', item.get('question_id')),
                    'question': item.get('question', item.get('text')),
                    'answer': item.get('answer'),
                    'type': item.get('type'),  # RHT or MHT
                    'category': item.get('category'),
                    'metadata': item
                })

            logger.info(f"Loaded {len(questions)} questions from Hugging Face")
            return questions

        except Exception as e:
            logger.error(f"Error loading from Hugging Face: {e}")
            raise

    def _load_from_local(self, split: str) -> List[Dict[str, Any]]:
        """Load from local files."""
        logger.info(f"Loading Med-HALT from local path: {self.dataset_path}")

        questions = []

        # Try split-specific file
        split_file = self.dataset_path / f"{split}.jsonl"
        if split_file.exists():
            questions = self._load_jsonl(split_file)
        else:
            # Try loading all JSONL files in directory
            for jsonl_file in self.dataset_path.glob("*.jsonl"):
                questions.extend(self._load_jsonl(jsonl_file))

        logger.info(f"Loaded {len(questions)} questions from local files")
        return questions

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load questions from JSONL file."""
        questions = []

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                questions.append({
                    'id': item.get('id', item.get('question_id')),
                    'question': item.get('question', item.get('text')),
                    'answer': item.get('answer'),
                    'type': item.get('type'),
                    'category': item.get('category'),
                    'metadata': item
                })

        return questions

    def get_rht_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for Reasoning Hallucination (RHT) questions."""
        return [q for q in questions if q.get('type') == 'RHT']

    def get_mht_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for Memory Hallucination (MHT) questions."""
        return [q for q in questions if q.get('type') == 'MHT']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Med-HALT dataset")
    parser.add_argument("--local-path", type=str, help="Local dataset path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--show-sample", action="store_true", help="Show sample questions")

    args = parser.parse_args()

    # Load dataset
    if args.local_path:
        loader = MedHALTLoader(dataset_path=Path(args.local_path), use_huggingface=False)
    else:
        loader = MedHALTLoader(use_huggingface=True)

    questions = loader.load(split=args.split)

    print(f"\nLoaded {len(questions)} questions")

    # Show breakdown
    rht = loader.get_rht_questions(questions)
    mht = loader.get_mht_questions(questions)
    print(f"  RHT (Reasoning): {len(rht)}")
    print(f"  MHT (Memory):    {len(mht)}")

    # Show sample
    if args.show_sample and questions:
        print("\nSample questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"\n{i}. ID: {q['id']}")
            print(f"   Type: {q.get('type', 'Unknown')}")
            print(f"   Question: {q['question'][:100]}...")
