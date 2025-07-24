"""Command-line interface for xdf2bids."""
import argparse
import logging
import sys
from pathlib import Path

from xdf2bids.xdf_processor import process_xdf_file, __version__

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def validate_arguments(input_file: str, output_dir: str):
    """Validate command line arguments."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    if not input_path.suffix.lower() == '.xdf':
        raise ValueError(f"Input file must be an XDF file: {input_file}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

def main():
    """Main function to handle command-line arguments and process XDF files."""
    parser = argparse.ArgumentParser(
        description="Convert XDF files to BIDS format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.xdf ./output/
  %(prog)s --verbose data.xdf ./bids_output/
        """
    )
    parser.add_argument("input_file", type=str, help="Path to the input XDF file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the BIDS formatted output.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Validate arguments
        validate_arguments(args.input_file, args.output_dir)
        
        logger.info(f"Processing XDF file: {args.input_file}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Process the file
        result = process_xdf_file(args.input_file, args.output_dir)
        
        if result:
            logger.info("XDF file processed successfully!")
        else:
            logger.error("Failed to process XDF file")
            sys.exit(1)
            
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()