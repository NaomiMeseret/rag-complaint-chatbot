"""Custom exceptions for the complaint analyzer."""


class ComplaintAnalyzerError(Exception):
    """Base exception for all complaint analyzer errors."""
    pass


class ConfigurationError(ComplaintAnalyzerError):
    """Raised when there's a configuration issue."""
    pass


class DataProcessingError(ComplaintAnalyzerError):
    """Raised when data processing fails."""
    pass


class VectorStoreError(ComplaintAnalyzerError):
    """Raised when vector store operations fail."""
    pass


class EmbeddingError(ComplaintAnalyzerError):
    """Raised when embedding generation fails."""
    pass


class ChunkingError(DataProcessingError):
    """Raised when text chunking fails."""
    pass


class RAGError(ComplaintAnalyzerError):
    """Raised when RAG pipeline operations fail."""
    pass

