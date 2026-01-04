"""ChromaDB vector store management."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

from ..utils.exceptions import VectorStoreError
from ..config import get_config

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages ChromaDB vector store for complaint embeddings."""
    
    def __init__(self, persist_path: Optional[str] = None, collection_name: str = "complaints", config=None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_path: Path to persist vector store
            collection_name: Name of collection
            config: Config object (optional)
        """
        if chromadb is None:
            raise VectorStoreError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        if config:
            persist_path = config.get('vector_store.path', persist_path) if persist_path is None else persist_path
            collection_name = config.get('vector_store.collection_name', collection_name)
            persist = config.get('vector_store.persist', True)
        else:
            persist = True
        
        try:
            if persist_path:
                persist_path = Path(persist_path)
                persist_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Initializing ChromaDB with persistence at {persist_path}")
                # Use PersistentClient if available, otherwise Client with Settings
                try:
                    self.client = chromadb.PersistentClient(path=str(persist_path))
                except AttributeError:
                    # Fallback for older/newer ChromaDB versions
                    self.client = chromadb.Client(Settings(persist_directory=str(persist_path)))
            else:
                logger.info("Initializing ChromaDB in-memory")
                self.client = chromadb.Client()
            
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"Connected to collection: {collection_name}")
        
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}") from e
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add embeddings and documents to vector store.
        
        Args:
            embeddings: Numpy array of embeddings
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (generated if not provided)
            
        Raises:
            VectorStoreError: If addition fails
        """
        try:
            if len(embeddings) != len(documents) != len(metadatas):
                raise VectorStoreError(
                    f"Mismatch in lengths: embeddings={len(embeddings)}, "
                    f"documents={len(documents)}, metadatas={len(metadatas)}"
                )
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Convert numpy array to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                batch_embeddings = embeddings_list[i:batch_end]
                batch_documents = documents[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]
                
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
        
        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            raise VectorStoreError(f"Error adding embeddings to vector store: {str(e)}") from e
    
    def add_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'chunk_text',
        embedding_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        id_column: Optional[str] = None
    ) -> None:
        """
        Add documents from DataFrame to vector store.
        
        Args:
            df: DataFrame with documents
            text_column: Column name with document texts
            embedding_column: Column name with pre-computed embeddings (optional)
            metadata_columns: Columns to include as metadata
            id_column: Column to use as IDs
            
        Raises:
            VectorStoreError: If addition fails
        """
        try:
            if text_column not in df.columns:
                raise VectorStoreError(f"Text column '{text_column}' not found in DataFrame")
            
            documents = df[text_column].tolist()
            
            # Prepare metadata
            if metadata_columns is None:
                metadata_columns = [col for col in df.columns if col not in [text_column, embedding_column, id_column]]
            
            metadatas = []
            for _, row in df.iterrows():
                metadata = {col: str(row[col]) for col in metadata_columns if col in row}
                metadatas.append(metadata)
            
            # Get IDs
            if id_column and id_column in df.columns:
                ids = df[id_column].astype(str).tolist()
            else:
                ids = None
            
            # Get embeddings if provided, otherwise will need to generate
            if embedding_column and embedding_column in df.columns:
                embeddings = np.array(df[embedding_column].tolist())
            else:
                raise VectorStoreError("Embeddings must be provided (embedding_column or pre-computed)")
            
            self.add_embeddings(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            raise VectorStoreError(f"Error adding DataFrame to vector store: {str(e)}") from e
    
    def search(
        self,
        query_embeddings: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query_embeddings: Query embedding(s)
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with results
        """
        try:
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            
            query_list = query_embeddings.tolist()
            
            results = self.collection.query(
                query_embeddings=query_list,
                n_results=n_results,
                where=where
            )
            
            return results
        
        except Exception as e:
            raise VectorStoreError(f"Error searching vector store: {str(e)}") from e
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count
            }
        except Exception as e:
            raise VectorStoreError(f"Error getting collection info: {str(e)}") from e

