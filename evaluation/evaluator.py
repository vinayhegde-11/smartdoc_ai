"""
RAG Evaluator using Ragas library and custom metrics.
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os

from ragas import evaluate, RunConfig
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from config.models import LLM_MODEL, EVAL_LLM_TEMPERATURE, VERTEX_EMBEDDING_MODEL, EVAL_TOP_K

from utils.logger import get_logger

logger = get_logger("evaluator")


class RAGEvaluator:
    """
    Comprehensive RAG evaluation using Ragas metrics and performance tracking.
    """
    
    def __init__(self, retrieval_pipeline, vector_store):
        """
        Initialize the evaluator.
        
        Args:
            retrieval_pipeline: The retrieval pipeline to evaluate
            vector_store: Vector store for document retrieval
        """
        self.retrieval_pipeline = retrieval_pipeline
        self.vector_store = vector_store
        
        # Initialize LLM for Ragas evaluation and answer generation
        self.llm = ChatVertexAI(
            model=LLM_MODEL,
            temperature=EVAL_LLM_TEMPERATURE  # Optimized for focused, relevant answers
        )
        
        # Initialize embeddings for Ragas
        self.embeddings = VertexAIEmbeddings(
            model_name=VERTEX_EMBEDDING_MODEL
        )
        
        # Wrap LLM for Ragas
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        
        # Metrics to evaluate
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]
        
        # Store evaluation history
        self.evaluation_history = []
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The query/question
            answer: The generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer for comparison
        
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Prepare data for Ragas
            data = {
                'question': [question],
                'answer': [answer],
                'contexts': [contexts],
            }
            
            if ground_truth:
                data['ground_truth'] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics_to_use = [
                context_precision,
                faithfulness,
                answer_relevancy
            ]
            
            if ground_truth:
                metrics_to_use.append(context_recall)
            
            # Configure Ragas with Vertex AI
            run_config = RunConfig(
                max_workers=4,
                max_wait=180
            )
            
            # Run evaluation in thread pool to avoid async conflicts
            def run_ragas_evaluation():
                """Run Ragas in a separate thread to avoid event loop conflicts."""
                return evaluate(
                    dataset,
                    metrics=metrics_to_use,
                    llm=self.ragas_llm,
                    embeddings=self.embeddings,
                    run_config=run_config
                )
            
            # Execute in thread pool
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_ragas_evaluation)
                result = future.result(timeout=300)  # 5 min timeout
            
            # Convert to dict and extract scores
            # Ragas returns EvaluationResult - access directly using keys
            import numpy as np
            
            scores = {
                'context_precision': float(np.mean(result['context_precision'])) if len(result['context_precision']) > 0 else None,
                'context_recall': float(np.mean(result['context_recall'])) if len(result['context_recall']) > 0 else None,
                'faithfulness': float(np.mean(result['faithfulness'])) if len(result['faithfulness']) > 0 else None,
                'answer_relevancy': float(np.mean(result['answer_relevancy'])) if len(result['answer_relevancy']) > 0 else None,
            }
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in single evaluation: {e}")
            return {
                'error': str(e),
                'context_precision': None,
                'context_recall': None,
                'faithfulness': None,
                'answer_relevancy': None
            }
    
    def evaluate_batch(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_dataset: List of test cases with question, ground_truth, etc.
        
        Returns:
            Comprehensive evaluation results including aggregated metrics
        """
        try:
            logger.info(f"Starting batch evaluation with {len(test_dataset)} test cases")
            
            results = []
            questions = []
            answers = []
            contexts_list = []
            ground_truths = []
            retrieval_times = []
            generation_times = []
            
            # Process each test case
            for idx, test_case in enumerate(test_dataset):
                question = test_case['question']
                ground_truth = test_case.get('ground_truth', '')
                
                logger.info(f"Evaluating test case {idx + 1}/{len(test_dataset)}: {question[:50]}...")
                
                # Measure retrieval time
                retrieval_start = time.time()
                retrieval_result = self.retrieval_pipeline.run_retrieval(
                    query=question,
                    top_k=EVAL_TOP_K  # Increased from 10 for better context precision
                )
                retrieval_time = time.time() - retrieval_start
                
                retrieved_docs = retrieval_result.get('results', [])
                contexts = [doc['text'] for doc in retrieved_docs]
                
                # Measure generation time
                generation_start = time.time()
                # Generate answer using the LLM with optimized prompt
                context_str = "\n\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])
                
                # Optimized prompt for better answer relevancy
                prompt = f"""Answer the question directly and concisely using ONLY the information from the context below.

Requirements:
- Provide a direct, focused answer (2-3 sentences maximum)
- Do not add preambles like "Based on the context..." 
- Do not include information not in the context
- Cite sources using [1], [2], etc. when referencing context

CONTEXT:
{context_str}

QUESTION: {question}

DIRECT ANSWER:"""
                
                # Use optimized generation parameters
                response = self.llm.invoke(
                    prompt,
                    max_tokens=150,  # Limit length for conciseness
                    temperature=EVAL_LLM_TEMPERATURE   # Low temperature for focused, relevant answers
                )
                answer = response.content
                generation_time = time.time() - generation_start
                
                # Store for batch evaluation
                questions.append(question)
                answers.append(answer)
                contexts_list.append(contexts)
                ground_truths.append(ground_truth)
                retrieval_times.append(retrieval_time)
                generation_times.append(generation_time)
                
                # Store individual result
                results.append({
                    'question': question,
                    'answer': answer,
                    'ground_truth': ground_truth,
                    'contexts': contexts,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time
                })
            
            # Prepare dataset for Ragas
            ragas_data = {
                'question': questions,
                'answer': answers,
                'contexts': contexts_list,
                'ground_truth': ground_truths
            }
            
            dataset = Dataset.from_dict(ragas_data)
            
            # Configure Ragas with Vertex AI
            run_config = RunConfig(
                max_workers=4,
                max_wait=300
            )
            
            # Run Ragas evaluation in a thread pool to avoid async conflicts
            logger.info("Running Ragas evaluation...")
            
            def run_ragas_evaluation():
                """Run Ragas in a separate thread to avoid event loop conflicts."""
                return evaluate(
                    dataset,
                    metrics=self.metrics,
                    llm=self.ragas_llm,
                    embeddings=self.embeddings,
                    run_config=run_config
                )
            
            # Execute in thread pool
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_ragas_evaluation)
                ragas_result = future.result(timeout=600)  # 10 min timeout
            
            logger.info(f"Ragas evaluation completed. Result type: {type(ragas_result)}")
            logger.info(f"Ragas result content: {ragas_result}")
            
            # Calculate aggregate metrics
            # Ragas returns an EvaluationResult object - access directly using keys
            try:
                import numpy as np
                
                # Access metrics directly from EvaluationResult object with try-except
                try:
                    cp_list = ragas_result['context_precision']
                except (KeyError, TypeError):
                    cp_list = []
                
                try:
                    cr_list = ragas_result['context_recall']
                except (KeyError, TypeError):
                    cr_list = []
                
                try:
                    f_list = ragas_result['faithfulness']
                except (KeyError, TypeError):
                    f_list = []
                
                try:
                    ar_list = ragas_result['answer_relevancy']
                except (KeyError, TypeError):
                    ar_list = []
                
                logger.info(f"Context Precision list: {cp_list}")
                logger.info(f"Context Recall list: {cr_list}")
                logger.info(f"Faithfulness list: {f_list}")
                logger.info(f"Answer Relevancy list: {ar_list}")
                
                avg_metrics = {
                    'context_precision': float(np.mean(cp_list)) if len(cp_list) > 0 else 0.0,
                    'context_recall': float(np.mean(cr_list)) if len(cr_list) > 0 else 0.0,
                    'faithfulness': float(np.mean(f_list)) if len(f_list) > 0 else 0.0,
                    'answer_relevancy': float(np.mean(ar_list)) if len(ar_list) > 0 else 0.0,
                    'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times),
                    'avg_generation_time': sum(generation_times) / len(generation_times)
                }
                logger.info(f"Aggregate metrics calculated: {avg_metrics}")
            except Exception as e:
                logger.error(f"Error calculating aggregate metrics: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Create evaluation report
            evaluation_report = {
                'timestamp': datetime.now().isoformat(),
                'num_test_cases': len(test_dataset),
                'aggregate_metrics': avg_metrics,
                'individual_results': results,
                'ragas_details': {
                    'context_precision': float(np.mean(cp_list)) if len(cp_list) > 0 else 0.0,
                    'context_recall': float(np.mean(cr_list)) if len(cr_list) > 0 else 0.0,
                    'faithfulness': float(np.mean(f_list)) if len(f_list) > 0 else 0.0,
                    'answer_relevancy': float(np.mean(ar_list)) if len(ar_list) > 0 else 0.0,
                }
            }
            
            # Store in history
            self.evaluation_history.append(evaluation_report)
            
            # Save to file
            self._save_evaluation_results(evaluation_report)
            
            logger.info("Batch evaluation completed successfully")
            return evaluation_report
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            raise
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        try:
            results_dir = "evaluation_results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results_dir}/evaluation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluation metrics from history.
        
        Returns:
            Dictionary with metrics summary and trends
        """
        if not self.evaluation_history:
            return {
                'message': 'No evaluation history available',
                'history': []
            }
        
        # Get latest evaluation
        latest = self.evaluation_history[-1]
        
        # Calculate trends if multiple evaluations exist
        trends = {}
        if len(self.evaluation_history) > 1:
            previous = self.evaluation_history[-2]
            for metric in ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']:
                current_val = latest['aggregate_metrics'].get(metric, 0)
                previous_val = previous['aggregate_metrics'].get(metric, 0)
                if previous_val > 0:
                    change = ((current_val - previous_val) / previous_val) * 100
                    trends[metric] = {
                        'change_percent': round(change, 2),
                        'direction': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                    }
        
        return {
            'latest_evaluation': latest,
            'total_evaluations': len(self.evaluation_history),
            'trends': trends,
            'history': [
                {
                    'timestamp': eval_result['timestamp'],
                    'metrics': eval_result['aggregate_metrics']
                }
                for eval_result in self.evaluation_history
            ]
        }
