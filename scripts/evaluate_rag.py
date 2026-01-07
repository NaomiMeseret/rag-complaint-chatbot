"""Evaluate RAG pipeline with test questions."""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_complaint_analyzer.rag import RAGPipeline
from src.rag_complaint_analyzer.utils.logger import setup_logger

logger = setup_logger()


def create_evaluation_questions():
    """Create a list of representative test questions."""
    return [
        {
            'id': 1,
            'question': "Why are people unhappy with Credit Cards?",
            'category': 'Product Analysis'
        },
        {
            'id': 2,
            'question': "What are the most common issues reported for Personal Loans?",
            'category': 'Issue Identification'
        },
        {
            'id': 3,
            'question': "What billing disputes have been reported recently?",
            'category': 'Specific Issue'
        },
        {
            'id': 4,
            'question': "How do complaints about Money Transfers compare to Credit Cards?",
            'category': 'Product Comparison'
        },
        {
            'id': 5,
            'question': "What fraud-related complaints have been filed?",
            'category': 'Fraud Detection'
        },
        {
            'id': 6,
            'question': "What are customers saying about Savings Accounts?",
            'category': 'Product Sentiment'
        },
        {
            'id': 7,
            'question': "What communication issues are customers experiencing?",
            'category': 'Service Quality'
        },
        {
            'id': 8,
            'question': "Which companies have the most complaints?",
            'category': 'Company Analysis'
        },
        {
            'id': 9,
            'question': "What are the main problems with account access?",
            'category': 'Technical Issues'
        },
        {
            'id': 10,
            'question': "What are customers complaining about regarding interest rates?",
            'category': 'Pricing Concerns'
        }
    ]


def evaluate_rag_pipeline(embeddings_path: str, output_file: str = "reports/evaluation_results.md"):
    """
    Evaluate RAG pipeline with test questions.
    
    Args:
        embeddings_path: Path to pre-built embeddings parquet file
        output_file: Path to save evaluation results
    """
    logger.info("Starting RAG pipeline evaluation...")
    
    # Initialize pipeline
    try:
        pipeline = RAGPipeline(embeddings_path=embeddings_path, retriever_top_k=5)
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Get test questions
    questions = create_evaluation_questions()
    
    # Run evaluation
    results = []
    for q in questions:
        logger.info(f"Evaluating question {q['id']}: {q['question']}")
        
        try:
            response = pipeline.query(q['question'], return_sources=True)
            
            # Extract source previews
            source_previews = []
            for i, source in enumerate(response['sources'][:2], 1):  # Show top 2 sources
                doc_preview = source['document'][:200] + "..." if len(source['document']) > 200 else source['document']
                source_previews.append(
                    f"Source {i} (Similarity: {source['similarity']:.3f}):\n"
                    f"  Product: {source['metadata'].get('product', 'N/A')}\n"
                    f"  Issue: {source['metadata'].get('issue', 'N/A')}\n"
                    f"  Preview: {doc_preview}"
                )
            
            results.append({
                'id': q['id'],
                'question': q['question'],
                'category': q['category'],
                'generated_answer': response['answer'],
                'retrieved_sources': "\n\n".join(source_previews),
                'quality_score': None,  # To be filled manually
                'comments': None  # To be filled manually
            })
            
            logger.info(f"Question {q['id']} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing question {q['id']}: {e}")
            results.append({
                'id': q['id'],
                'question': q['question'],
                'category': q['category'],
                'generated_answer': f"ERROR: {str(e)}",
                'retrieved_sources': "N/A",
                'quality_score': None,
                'comments': f"Processing failed: {str(e)}"
            })
    
    # Generate evaluation report
    generate_evaluation_report(results, output_file)
    
    logger.info(f"Evaluation complete. Results saved to {output_file}")
    print(f"\n✓ Evaluation complete! Results saved to {output_file}")
    print(f"  - Evaluated {len(results)} questions")
    print(f"  - Successful: {sum(1 for r in results if 'ERROR' not in r['generated_answer'])}")
    print(f"  - Failed: {sum(1 for r in results if 'ERROR' in r['generated_answer'])}")


def generate_evaluation_report(results: list, output_file: str):
    """Generate markdown evaluation report."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# RAG Pipeline Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"This report contains the evaluation of the RAG pipeline using {len(results)} test questions.",
        "",
        "## Evaluation Methodology",
        "",
        "- **Retrieval:** Top-5 most similar chunks retrieved using cosine similarity",
        "- **Generation:** LLM-based answer generation from retrieved context",
        "- **Evaluation:** Manual quality assessment (1-5 scale)",
        "",
        "## Results",
        "",
        "| Question ID | Category | Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |",
        "|------------|----------|----------|------------------|-------------------|---------------|-------------------|",
    ]
    
    for result in results:
        # Truncate long answers for table
        answer_preview = result['generated_answer'][:200] + "..." if len(result['generated_answer']) > 200 else result['generated_answer']
        sources_preview = result['retrieved_sources'].replace('\n', ' ')[:150] + "..." if len(result['retrieved_sources']) > 150 else result['retrieved_sources'].replace('\n', ' ')
        
        # Escape pipe characters for markdown table
        answer_preview = answer_preview.replace('|', '\\|')
        sources_preview = sources_preview.replace('|', '\\|')
        
        quality_score = result['quality_score'] if result['quality_score'] else "TBD"
        comments = result['comments'] if result['comments'] else "TBD"
        
        report_lines.append(
            f"| {result['id']} | {result['category']} | {result['question']} | {answer_preview} | {sources_preview} | {quality_score} | {comments} |"
        )
    
    report_lines.extend([
        "",
        "## Detailed Results",
        ""
    ])
    
    for result in results:
        report_lines.extend([
            f"### Question {result['id']}: {result['question']}",
            "",
            f"**Category:** {result['category']}",
            "",
            "**Generated Answer:**",
            "",
            result['generated_answer'],
            "",
            "**Retrieved Sources:**",
            "",
            result['retrieved_sources'],
            "",
            f"**Quality Score:** {result['quality_score'] if result['quality_score'] else 'TBD'}",
            "",
            f"**Comments/Analysis:** {result['comments'] if result['comments'] else 'TBD'}",
            "",
            "---",
            ""
        ])
    
    report_lines.extend([
        "## Summary",
        "",
        "### What Worked Well",
        "",
        "- [To be filled after manual review]",
        "",
        "### What Could Be Improved",
        "",
        "- [To be filled after manual review]",
        "",
        "### Next Steps",
        "",
        "- Complete manual quality scoring (1-5 scale)",
        "- Analyze patterns in successful vs. unsuccessful queries",
        "- Refine prompt template based on evaluation results",
        "- Consider fine-tuning retrieval parameters (top_k, similarity thresholds)",
        ""
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/raw/complaint_embeddings.parquet",
        help="Path to pre-built embeddings parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/evaluation_results.md",
        help="Path to save evaluation report"
    )
    
    args = parser.parse_args()
    
    evaluate_rag_pipeline(args.embeddings, args.output)

