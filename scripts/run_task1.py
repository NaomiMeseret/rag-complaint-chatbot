"""Script to run Task 1: Data preprocessing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_complaint_analyzer.utils.logger import setup_logger
from rag_complaint_analyzer.config import get_config
from rag_complaint_analyzer.data.preprocessing import DataPreprocessor

logger = setup_logger()


def main():
    """Run Task 1: EDA and preprocessing."""
    try:
        config = get_config()
        preprocessor = DataPreprocessor(config)
        
        # Input and output paths
        input_file = config.data_dir / "complaints.csv"
        output_file = Path(config.get('data.filtered_file'))
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info("Starting Task 1: Data Preprocessing")
        df = preprocessor.process_data(str(input_file), str(output_file))
        
        logger.info(f"Task 1 completed successfully. Processed {len(df)} complaints.")
        print(f"\n✓ Task 1 Complete!")
        print(f"  - Processed {len(df)} complaints")
        print(f"  - Output saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Task 1 failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

