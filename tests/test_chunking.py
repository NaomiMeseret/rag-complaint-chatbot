"""Tests for text chunking module."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_complaint_analyzer.processing.chunking import TextChunker
from rag_complaint_analyzer.utils.exceptions import ChunkingError


def test_chunk_text():
    """Test text chunking."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    # Test normal text
    text = "A" * 250  # 250 characters
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    
    # Test empty text
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text(None) == []


def test_chunk_dataframe():
    """Test DataFrame chunking."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    df = pd.DataFrame({
        'Complaint ID': [1, 2],
        'text': ['A' * 250, 'B' * 150],
        'Product': ['Credit card', 'Personal loan']
    })
    
    chunk_df = chunker.chunk_dataframe(df, text_column='text')
    assert len(chunk_df) > len(df)  # Should have more rows than input


if __name__ == "__main__":
    pytest.main([__file__])

