# XDF2BIDS

Reads .xdf files and returns data in a BIDS compliant format (.tsv for data and .json for metadata). It was designed to handle motion capture data from kinectv2, the wii balance board as well as eye tracking data. It returns the data for largest possible overlapping time window where all channels were recorded.
It provides a separate json sidecar for each data file as well as a separate events file containing data from all detected marker streams.

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
