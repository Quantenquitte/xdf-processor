"""IDE interface for xdf2bids."""

from .xdf_processor import process_xdf_file, __version__

INPUT_FILES = ["test_file.xdf"]  # Example input files, replace with actual paths
OUTPUT_DIR = None

if __name__ == "__main__":
    """Main function to process XDF files."""
    for input_file in INPUT_FILES:
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output"
        print(f"Processing {input_file} to {OUTPUT_DIR}")
        process_xdf_file(input_file, OUTPUT_DIR)
