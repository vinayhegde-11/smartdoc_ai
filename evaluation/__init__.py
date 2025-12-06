"""
Evaluation module for SmartDoc AI.

This module provides RAG evaluation capabilities using the Ragas library
and custom performance metrics.
"""

from .evaluator import RAGEvaluator
from .test_dataset import load_test_dataset, create_sample_dataset

__all__ = ['RAGEvaluator', 'load_test_dataset', 'create_sample_dataset']
