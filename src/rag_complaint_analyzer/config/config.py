"""Configuration management using YAML files."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the complaint analyzer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in the config directory
            base_dir = Path(__file__).parent
            config_path = base_dir / "config.yaml"
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'data.raw_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.get('data.raw_dir', 'data/raw'))
    
    @property
    def processed_dir(self) -> Path:
        """Get processed data directory path."""
        return Path(self.get('data.processed_dir', 'data/processed'))
    
    @property
    def vector_store_path(self) -> Path:
        """Get vector store path."""
        return Path(self.get('vector_store.path', 'vector_store'))
    
    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        return Path(self.get('logging.log_dir', 'logs'))


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance

