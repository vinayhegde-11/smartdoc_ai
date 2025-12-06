"""
Test dataset management for RAG evaluation.
"""

import json
import os
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger("test_dataset")


def load_test_dataset(filepath: str = "evaluation/sample_test_set.json") -> List[Dict[str, Any]]:
    """
    Load test dataset from JSON file.
    
    Args:
        filepath: Path to the test dataset JSON file
    
    Returns:
        List of test cases
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Test dataset not found at {filepath}. Creating sample dataset.")
            create_sample_dataset(filepath)
        
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} test cases from {filepath}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        return []


def create_sample_dataset(filepath: str = "evaluation/sample_test_set.json") -> List[Dict[str, Any]]:
    """
    Create a sample test dataset for evaluation.
    
    Args:
        filepath: Path where to save the sample dataset
    
    Returns:
        List of sample test cases
    """
    sample_dataset = [
        {
            "question": "What is the main topic of this document?",
            "ground_truth": "The document discusses advanced retrieval-augmented generation techniques for improving question answering systems.",
            "expected_context_keywords": ["retrieval", "generation", "RAG"]
        },
        {
            "question": "What are the key benefits mentioned?",
            "ground_truth": "The key benefits include improved accuracy, better context utilization, and enhanced user experience.",
            "expected_context_keywords": ["benefits", "advantages", "improvements"]
        },
        {
            "question": "How does the system handle multi-turn conversations?",
            "ground_truth": "The system maintains conversation history and uses it to provide context-aware responses in subsequent turns.",
            "expected_context_keywords": ["conversation", "history", "multi-turn"]
        }
    ]
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(sample_dataset, f, indent=2)
        
        logger.info(f"Created sample test dataset at {filepath}")
        return sample_dataset
    except Exception as e:
        logger.error(f"Error creating sample dataset: {e}")
        return sample_dataset


def validate_test_dataset(dataset: List[Dict[str, Any]]) -> bool:
    """
    Validate that a test dataset has the required format.
    
    Args:
        dataset: Test dataset to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not dataset or not isinstance(dataset, list):
        logger.error("Dataset must be a non-empty list")
        return False
    
    required_fields = ['question', 'ground_truth']
    
    for idx, test_case in enumerate(dataset):
        if not isinstance(test_case, dict):
            logger.error(f"Test case {idx} is not a dictionary")
            return False
        
        for field in required_fields:
            if field not in test_case:
                logger.error(f"Test case {idx} missing required field: {field}")
                return False
            
            if not test_case[field] or not isinstance(test_case[field], str):
                logger.error(f"Test case {idx} has invalid {field}")
                return False
    
    logger.info(f"Dataset validation successful ({len(dataset)} test cases)")
    return True


def export_evaluation_results(results: Dict[str, Any], filepath: str):
    """
    Export evaluation results to a file.
    
    Args:
        results: Evaluation results dictionary
        filepath: Path where to save the results
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Exported evaluation results to {filepath}")
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
