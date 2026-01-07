"""
RAG-Powered Complaint Analysis System
A system for analyzing customer complaints using Retrieval-Augmented Generation.
"""

__version__ = "0.1.0"
__author__ = "CrediTrust Financial Data & AI Team"

from .config import Config
from .utils.logger import setup_logger

__all__ = ["Config", "setup_logger"]

