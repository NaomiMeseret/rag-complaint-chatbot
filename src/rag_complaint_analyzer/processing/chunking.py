"""Text chunking utilities."""

import pandas as pd
from typing import List, Dict, Any
import logging

from ..utils.exceptions import ChunkingError
from ..config import get_config

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking for complaint narratives."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, config=None):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            config: Config object (optional)
        """
        if config:
            self.chunk_size = config.get('text_processing.chunk_size', chunk_size)
            self.chunk_overlap = config.get('text_processing.chunk_overlap', chunk_overlap)
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        logger.info(f"Initialized TextChunker: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        try:
            if not text or len(text.strip()) == 0:
                return []
            
            text = str(text)
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
                
                if end >= len(text):
                    break
                
                start += self.chunk_size - self.chunk_overlap
            
            return chunks
        
        except Exception as e:
            raise ChunkingError(f"Error chunking text: {str(e)}") from e
    
    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str = 'Complaint ID',
        metadata_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Chunk all texts in a DataFrame and create chunk records.
        
        Args:
            df: DataFrame with text to chunk
            text_column: Column name containing text
            id_column: Column name for unique identifier
            metadata_columns: Additional columns to include in output
            
        Returns:
            DataFrame with chunk records
        """
        if metadata_columns is None:
            metadata_columns = ['Product', 'Issue', 'Company', 'State', 'Date received']
        
        try:
            logger.info(f"Chunking {len(df)} records from column '{text_column}'")
            
            chunk_records = []
            
            for idx, row in df.iterrows():
                text = row[text_column]
                chunks = self.chunk_text(text)
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    record = {
                        'complaint_id': row[id_column],
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'chunk_text': chunk_text
                    }
                    
                    # Add metadata columns
                    for col in metadata_columns:
                        if col in row:
                            record[col.lower().replace(' ', '_')] = row[col]
                    
                    chunk_records.append(record)
            
            chunk_df = pd.DataFrame(chunk_records)
            logger.info(f"Created {len(chunk_df)} chunks from {len(df)} records")
            
            return chunk_df
        
        except Exception as e:
            if isinstance(e, ChunkingError):
                raise
            raise ChunkingError(f"Error chunking DataFrame: {str(e)}") from e

