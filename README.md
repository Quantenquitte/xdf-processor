# XDF2BIDS

Convert multi-dimensional, cross-devices .xdf data files to BIDS compliant format.

## Installation

```bash
clone the repository
pip install -e .
```
*Note: If you are using python <3.8 you will get an error message from the pyxdf module. Follow the instructions outlined in the terminal.

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
