"""Stratified sampling utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional
import logging

from ..utils.exceptions import DataProcessingError
from ..config import get_config

logger = logging.getLogger(__name__)


class StratifiedSampler:
    """Handles stratified sampling of complaint data."""
    
    def __init__(self, sample_size: int = 12000, random_state: int = 42, config=None):
        """
        Initialize stratified sampler.
        
        Args:
            sample_size: Target sample size
            random_state: Random seed for reproducibility
            config: Config object (optional)
        """
        if config:
            self.sample_size = config.get('sampling.sample_size', sample_size)
            self.random_state = config.get('sampling.random_state', random_state)
            self.stratify_by = config.get('sampling.stratify_by', 'Product')
        else:
            self.sample_size = sample_size
            self.random_state = random_state
            self.stratify_by = 'Product'
        
        logger.info(f"Initialized StratifiedSampler: sample_size={self.sample_size}, random_state={self.random_state}")
    
    def sample(
        self,
        df: pd.DataFrame,
        stratify_column: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create stratified sample from DataFrame.
        
        Args:
            df: Input DataFrame
            stratify_column: Column to stratify by (defaults to config value)
            output_file: Path to save sample (optional)
            
        Returns:
            Sampled DataFrame
            
        Raises:
            DataProcessingError: If sampling fails
        """
        try:
            if stratify_column is None:
                stratify_column = self.stratify_by
            
            if stratify_column not in df.columns:
                raise DataProcessingError(f"Stratify column '{stratify_column}' not found in DataFrame")
            
            # Adjust sample size if dataset is smaller
            max_samples = min(self.sample_size, len(df))
            
            logger.info(f"Sampling {max_samples} records from {len(df)} total records, stratified by '{stratify_column}'")
            
            if max_samples >= len(df):
                logger.warning(f"Sample size ({max_samples}) >= dataset size ({len(df)}). Returning full dataset.")
                sampled_df = df.copy()
            else:
                sampled_df, _ = train_test_split(
                    df,
                    train_size=max_samples,
                    stratify=df[stratify_column],
                    random_state=self.random_state
                )
            
            # Reset index
            sampled_df = sampled_df.reset_index(drop=True)
            
            # Log distribution
            distribution = sampled_df[stratify_column].value_counts(normalize=True)
            logger.info(f"Sample distribution by {stratify_column}:")
            for product, prop in distribution.items():
                logger.info(f"  {product}: {prop:.2%}")
            
            # Save if output path provided
            if output_file:
                sampled_df.to_csv(output_file, index=False)
                logger.info(f"Saved sample to {output_file}")
            
            return sampled_df
        
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Error in stratified sampling: {str(e)}") from e

