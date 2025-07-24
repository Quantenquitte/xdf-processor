"""
XDF to BIDS Converter Package

Converts XDF files to Brain Imaging Data Structure (BIDS) format.
"""

__version__ = "1.0.0"
__author__ = "XDF Ecosystem Team"

from .xdf_processor import XDFProcessor, process_xdf_file

__all__ = [
    'XDFProcessor',
    'process_xdf_file',
    '__version__'
]