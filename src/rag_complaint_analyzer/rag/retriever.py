"""Retriever implementation for RAG pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..vector_store.embedder import Embedder
from ..utils.exceptions import RAGError

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks based on query similarity."""
    
    def __init__(
        self,
        embeddings_path: str,
        embedder: Optional[Embedder] = None,
        top_k: int = 5
    ):
        """
        Initialize retriever.
        
        Args:
            embeddings_path: Path to parquet file with pre-built embeddings
            embedder: Embedder instance for query embeddings
            top_k: Number of top results to retrieve
        """
        self.embeddings_path = Path(embeddings_path)
        if not self.embeddings_path.exists():
            raise RAGError(f"Embeddings file not found: {embeddings_path}")
        
        self.embedder = embedder
        if embedder is None:
            from ..vector_store.embedder import Embedder
            self.embedder = Embedder()
        
        self.top_k = top_k
        self._embeddings_df = None
        self._embeddings_matrix = None
        
        logger.info(f"Initializing retriever with embeddings from {embeddings_path}")
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings from parquet file."""
        try:
            logger.info("Loading pre-built embeddings...")
            self._embeddings_df = pd.read_parquet(self.embeddings_path)
            logger.info(f"Loaded {len(self._embeddings_df)} document embeddings")
            
            # Convert embeddings to numpy matrix for efficient similarity search
            embeddings_list = self._embeddings_df['embedding'].tolist()
            if isinstance(embeddings_list[0], np.ndarray):
                self._embeddings_matrix = np.vstack(embeddings_list)
            else:
                # Convert list of lists to numpy array
                self._embeddings_matrix = np.array(embeddings_list)
            
            logger.info(f"Embeddings matrix shape: {self._embeddings_matrix.shape}")
            
        except Exception as e:
            raise RAGError(f"Failed to load embeddings: {str(e)}") from e
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant document chunks for a query.
        
        Args:
            query: User question/query string
            top_k: Number of results to return (overrides default)
            
        Returns:
            List of dictionaries with retrieved chunks and metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Embed the query
            logger.info(f"Embedding query: {query[:100]}...")
            query_embedding = self.embedder.embed(query)
            
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Compute cosine similarity
            # Normalize for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
            embeddings_norm = self._embeddings_matrix / (np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8)
            
            # Compute similarities
            similarities = np.dot(embeddings_norm, query_norm.T).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                row = self._embeddings_df.iloc[idx]
                result = {
                    'document': row['document'],
                    'id': row['id'],
                    'metadata': row['metadata'],
                    'similarity': float(similarities[idx])
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            raise RAGError(f"Error retrieving documents: {str(e)}") from e
    
    def retrieve_with_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve and format chunks as context string.
        
        Args:
            query: User question/query string
            top_k: Number of results to return
            
        Returns:
            Formatted context string from retrieved chunks
        """
        results = self.retrieve(query, top_k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            context_parts.append(
                f"[Chunk {i}] "
                f"Product: {metadata.get('product', 'N/A')}, "
                f"Issue: {metadata.get('issue', 'N/A')}, "
                f"Company: {metadata.get('company', 'N/A')}\n"
                f"{result['document']}\n"
            )
        
        return "\n".join(context_parts)

