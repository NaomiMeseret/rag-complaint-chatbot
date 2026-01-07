"""Tests for data preprocessing module."""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_complaint_analyzer.data.preprocessing import DataPreprocessor
from rag_complaint_analyzer.config import Config
from rag_complaint_analyzer.utils.exceptions import DataProcessingError


def test_clean_text():
    """Test text cleaning function."""
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # Test basic cleaning
    text = "I am writing to file a complaint. This is my ISSUE!"
    cleaned = preprocessor.clean_text(text)
    assert cleaned.islower()
    assert "complaint" in cleaned
    
    # Test empty text
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""


def test_filter_data():
    """Test data filtering."""
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # Create test data
    df = pd.DataFrame({
        'Product': ['Credit card', 'Debt collection', 'Credit card', 'Personal loan'],
        'Consumer complaint narrative': ['Text 1', 'Text 2', '', 'Text 4']
    })
    
    filtered = preprocessor.filter_data(df)
    
    # Should only include target products with non-empty narratives
    assert len(filtered) > 0
    assert all(filtered['Product'].isin(preprocessor.target_products))


if __name__ == "__main__":
    pytest.main([__file__])

