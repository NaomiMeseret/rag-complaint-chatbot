"""Complete RAG pipeline combining retriever and generator."""

import logging
from typing import Dict, Any, List, Optional

from .retriever import Retriever
from .generator import Generator
from ..utils.exceptions import RAGError

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for complaint analysis."""
    
    def __init__(
        self,
        embeddings_path: str,
        embedder=None,
        retriever_top_k: int = 5,
        generator_model: Optional[str] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embeddings_path: Path to pre-built embeddings parquet file
            embedder: Embedder instance (optional)
            retriever_top_k: Number of chunks to retrieve
            generator_model: LLM model name (optional)
            prompt_template: Custom prompt template (optional)
        """
        logger.info("Initializing RAG pipeline...")
        
        self.retriever = Retriever(
            embeddings_path=embeddings_path,
            embedder=embedder,
            top_k=retriever_top_k
        )
        
        self.generator = Generator(
            model_name=generator_model,
            prompt_template=prompt_template
        )
        
        logger.info("RAG pipeline initialized successfully")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user question through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Override default top_k for retrieval
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally sources
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
            
            if not retrieved_chunks:
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': []
                }
            
            # Format context
            context = self.retriever.retrieve_with_context(question, top_k=top_k)
            
            # Generate answer
            answer = self.generator.generate(question, context)
            
            # Prepare response
            response = {
                'answer': answer,
                'question': question
            }
            
            if return_sources:
                sources = [
                    {
                        'document': chunk['document'][:500] + "..." if len(chunk['document']) > 500 else chunk['document'],
                        'metadata': chunk['metadata'],
                        'similarity': chunk['similarity'],
                        'id': chunk['id']
                    }
                    for chunk in retrieved_chunks
                ]
                response['sources'] = sources
            
            logger.info("Question processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            raise RAGError(f"Failed to process question: {str(e)}") from e

