"""XDF to BIDS converter package."""

from .xdf_processor import __version__, process_xdf_file, XDFProcessor

__all__ = ["__version__", "process_xdf_file", "XDFProcessor"]