"""Data preprocessing and cleaning utilities."""

import pandas as pd
import re
from pathlib import Path
from typing import List, Optional
import logging

from ..utils.exceptions import DataProcessingError
from ..config import get_config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, cleaning, and filtering."""
    
    def __init__(self, config=None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Config object (optional, will use global config if not provided)
        """
        if config is None:
            config = get_config()
        self.config = config
        self.target_products = config.get('products', [])
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load complaint data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with complaint data
            
        Raises:
            DataProcessingError: If file cannot be loaded
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            raise DataProcessingError(f"Failed to load data from {file_path}: {str(e)}") from e
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize complaint narrative text.
        
        Args:
            text: Raw complaint text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower().strip()
        
        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r'i am writing to (file|lodge|submit) a complaint[\s\S]*?\.',
            r'dear (sir|madam|team)[\s\S]*?\.',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove special characters (keep alphanumeric, spaces, periods, commas)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def filter_data(
        self,
        df: pd.DataFrame,
        product_column: str = 'Product',
        narrative_column: str = 'Consumer complaint narrative'
    ) -> pd.DataFrame:
        """
        Filter data for target products and non-empty narratives.
        
        Args:
            df: Input DataFrame
            product_column: Name of product column
            narrative_column: Name of narrative column
            
        Returns:
            Filtered DataFrame
            
        Raises:
            DataProcessingError: If required columns are missing
        """
        try:
            # Check required columns
            if product_column not in df.columns:
                raise DataProcessingError(f"Column '{product_column}' not found in DataFrame")
            if narrative_column not in df.columns:
                raise DataProcessingError(f"Column '{narrative_column}' not found in DataFrame")
            
            # Filter for target products
            initial_count = len(df)
            df_filtered = df[df[product_column].isin(self.target_products)].copy()
            logger.info(f"Filtered to {len(df_filtered)} records for target products (from {initial_count})")
            
            # Remove empty narratives
            df_filtered = df_filtered[
                df_filtered[narrative_column].notna() &
                (df_filtered[narrative_column].astype(str).str.strip() != '')
            ].copy()
            
            logger.info(f"Filtered to {len(df_filtered)} records with non-empty narratives")
            
            return df_filtered.reset_index(drop=True)
        
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Error filtering data: {str(e)}") from e
    
    def process_data(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        clean_narrative_column: str = 'clean_narrative'
    ) -> pd.DataFrame:
        """
        Complete data processing pipeline: load, filter, and clean.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save processed data (optional)
            clean_narrative_column: Name for cleaned narrative column
            
        Returns:
            Processed DataFrame
        """
        try:
            # Load data
            df = self.load_data(input_file)
            
            # Filter data
            df = self.filter_data(df)
            
            # Clean narratives
            narrative_column = 'Consumer complaint narrative'
            if narrative_column in df.columns:
                logger.info("Cleaning complaint narratives...")
                df[clean_narrative_column] = df[narrative_column].apply(self.clean_text)
                logger.info("Narrative cleaning completed")
            
            # Save if output path provided
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed data to {output_path}")
            
            return df
        
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Error in data processing pipeline: {str(e)}") from e

