"""Script to run Task 2: Chunking, Embedding, and Vector Store."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from rag_complaint_analyzer.utils.logger import setup_logger
from rag_complaint_analyzer.config import get_config
from rag_complaint_analyzer.processing.sampling import StratifiedSampler
from rag_complaint_analyzer.processing.chunking import TextChunker
from rag_complaint_analyzer.vector_store.embedder import Embedder
from rag_complaint_analyzer.vector_store.chroma_store import ChromaVectorStore

logger = setup_logger()


def main():
    """Run Task 2: Chunking, Embedding, Vector Store."""
    try:
        config = get_config()
        
        logger.info("Starting Task 2: Chunking, Embedding, and Vector Store")
        
        # Load filtered data
        filtered_file = Path(config.get('data.filtered_file'))
        if not filtered_file.exists():
            raise FileNotFoundError(f"Filtered data not found: {filtered_file}. Run Task 1 first.")
        
        logger.info(f"Loading filtered data from {filtered_file}")
        df = pd.read_csv(filtered_file)
        
        # Step 1: Stratified sampling
        logger.info("Step 1: Creating stratified sample")
        sampler = StratifiedSampler(config=config)
        sample_df = sampler.sample(df, output_file=config.get('data.sample_file'))
        
        # Step 2: Chunking
        logger.info("Step 2: Chunking narratives")
        chunker = TextChunker(config=config)
        chunk_df = chunker.chunk_dataframe(
            sample_df,
            text_column='clean_narrative',
            metadata_columns=['Product', 'Issue', 'Company', 'State', 'Date received']
        )
        chunk_df.to_csv(config.get('data.chunked_file'), index=False)
        logger.info(f"Created {len(chunk_df)} chunks")
        
        # Step 3: Embedding
        logger.info("Step 3: Generating embeddings")
        embedder = Embedder(config=config)
        embeddings = embedder.embed(chunk_df['chunk_text'].tolist(), show_progress=True)
        
        # Step 4: Vector store
        logger.info("Step 4: Storing in vector database")
        vector_store = ChromaVectorStore(config=config)
        
        # Prepare metadata and IDs
        metadatas = []
        ids = []
        for _, row in chunk_df.iterrows():
            metadata = {
                'complaint_id': str(row.get('complaint_id', '')),
                'product': str(row.get('product', '')),
                'issue': str(row.get('issue', '')),
                'company': str(row.get('company', '')),
                'state': str(row.get('state', '')),
                'date_received': str(row.get('date_received', '')),
                'chunk_index': int(row.get('chunk_index', 0)),
                'total_chunks': int(row.get('total_chunks', 0))
            }
            metadatas.append(metadata)
            ids.append(f"{metadata['complaint_id']}_{metadata['chunk_index']}")
        
        vector_store.add_embeddings(
            embeddings=embeddings,
            documents=chunk_df['chunk_text'].tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        info = vector_store.get_collection_info()
        
        logger.info("Task 2 completed successfully")
        print(f"\n✓ Task 2 Complete!")
        print(f"  - Created {len(sample_df)} sample records")
        print(f"  - Generated {len(chunk_df)} chunks")
        print(f"  - Stored {info['document_count']} documents in vector store")
        print(f"  - Vector store location: {config.vector_store_path}")
    
    except Exception as e:
        logger.error(f"Task 2 failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

