"""Command-line interface for the complaint analyzer."""

import argparse
import sys
from pathlib import Path

from .utils.logger import setup_logger
from .config import get_config
from .data.preprocessing import DataPreprocessor
from .processing.sampling import StratifiedSampler
from .processing.chunking import TextChunker
from .vector_store.embedder import Embedder
from .vector_store.chroma_store import ChromaVectorStore

logger = setup_logger()


def process_data_cli(args):
    """CLI command for data processing."""
    try:
        config = get_config(args.config)
        preprocessor = DataPreprocessor(config)
        
        output_file = args.output or config.get('data.filtered_file')
        df = preprocessor.process_data(args.input, output_file)
        
        print(f"✓ Processed {len(df)} complaints")
        print(f"✓ Saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        sys.exit(1)


def build_vector_store_cli(args):
    """CLI command for building vector store."""
    try:
        config = get_config(args.config)
        
        # Load processed data
        import pandas as pd
        df = pd.read_csv(config.get('data.filtered_file'))
        
        # Sample
        sampler = StratifiedSampler(config=config)
        sample_df = sampler.sample(df, output_file=config.get('data.sample_file'))
        
        # Chunk
        chunker = TextChunker(config=config)
        chunk_df = chunker.chunk_dataframe(sample_df, text_column='clean_narrative')
        chunk_df.to_csv(config.get('data.chunked_file'), index=False)
        
        # Embed
        embedder = Embedder(config=config)
        embeddings = embedder.embed(chunk_df['chunk_text'].tolist())
        
        # Store
        vector_store = ChromaVectorStore(config=config)
        
        # Prepare metadata
        metadata_cols = ['complaint_id', 'product', 'issue', 'company', 'state', 'date_received', 'chunk_index', 'total_chunks']
        metadatas = []
        for _, row in chunk_df.iterrows():
            metadata = {col: str(row.get(col, '')) for col in metadata_cols if col in chunk_df.columns}
            metadatas.append(metadata)
        
        ids = [f"{row['complaint_id']}_{row['chunk_index']}" for _, row in chunk_df.iterrows()]
        
        vector_store.add_embeddings(
            embeddings=embeddings,
            documents=chunk_df['chunk_text'].tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        info = vector_store.get_collection_info()
        print(f"✓ Vector store built with {info['document_count']} documents")
    
    except Exception as e:
        logger.error(f"Error building vector store: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Complaint Analyzer CLI")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process data command
    process_parser = subparsers.add_parser("process", help="Process and clean complaint data")
    process_parser.add_argument("input", help="Input CSV file path")
    process_parser.add_argument("--output", help="Output CSV file path")
    
    # Build vector store command
    build_parser = subparsers.add_parser("build", help="Build vector store from processed data")
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_data_cli(args)
    elif args.command == "build":
        build_vector_store_cli(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

