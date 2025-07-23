# XDF2BIDS

Convert multi-dimensional, cross-devices .xdf data files to BIDS compliant format.

## Installation

```bash
clone the repository
navigate to the xdf_processor folder
pip install -e .
```

## Usage

### Command Line Interface

```bash
xdf2bids input_file.xdf output_directory/
```

### Python Module

```python
from xdf2bids import process_xdf_file

process_xdf_file("input_file.xdf", "output_directory/")
```
