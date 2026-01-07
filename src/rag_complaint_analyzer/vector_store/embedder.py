"""Text embedding generation using sentence transformers."""

import numpy as np
from typing import List, Union
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..utils.exceptions import EmbeddingError
from ..config import get_config

logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, config=None):
        """
        Initialize embedder with specified model.
        
        Args:
            model_name: Name of sentence transformer model
            batch_size: Batch size for embedding generation
            config: Config object (optional)
        """
        if SentenceTransformer is None:
            raise EmbeddingError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        
        if config:
            model_name = config.get('embeddings.model_name', model_name)
            batch_size = config.get('embeddings.batch_size', batch_size)
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.batch_size = batch_size
            self.model_name = model_name
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {str(e)}") from e
    
    def embed(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (2D if multiple texts, 1D if single text)
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                raise EmbeddingError("Empty text list provided")
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Error generating embeddings: {str(e)}") from e
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

