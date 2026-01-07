"""Script to run Task 3: RAG Pipeline Evaluation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_rag import evaluate_rag_pipeline
from src.rag_complaint_analyzer.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Run Task 3: RAG Pipeline Evaluation."""
    logger.info("Starting Task 3: RAG Pipeline Evaluation")
    
    try:
        # Path to pre-built embeddings
        embeddings_path = "data/raw/complaint_embeddings.parquet"
        
        # Output file
        output_file = "reports/evaluation_results.md"
        
        # Run evaluation
        evaluate_rag_pipeline(embeddings_path, output_file)
        
        logger.info("Task 3 completed successfully")
        print("\n✓ Task 3 Complete!")
        print(f"  - Evaluation results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Task 3 failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

