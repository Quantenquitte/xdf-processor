""" xdf_processor.py - XDF Processor for BIDS Export using shared xdf_core components
    Copyright (C) 2025 Janik Pawlowski

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__version__ = "1.0.0"

import logging
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Use shared components from xdf_core
try:
    from xdf_core import (
        XDFLoader, StreamClassifier, TimeUtils, 
        ChannelParser, BIDSExporter
    )
except ImportError:
    # Fallback for development - show helpful message
    print("ERROR: xdf_core not found. Please install with:")
    print("  cd /path/to/xdf_core && pip install -e .")
    print("\nOr install from the combined workspace:")
    print("  pip install -e xdf_core/")
    raise

logger = logging.getLogger(__name__)

SAVE_FOLDER = 'data/preprocessed'
WII_BOARD_WIDTH = 43.3  # cm
WII_BOARD_LENGTH = 23.8  # cm

class XDFProcessor:
    """XDF processor using shared xdf_core components for BIDS export"""

    def __init__(self, use_wii=True, **kwargs):
        """Initialize with shared components"""
        self.loader = XDFLoader()
        self.classifier = StreamClassifier()
        self.parser = ChannelParser()
        self.exporter = BIDSExporter()
        self.time_utils = TimeUtils()
        
        # Fix: Initialize Wii board dimensions
        self.WII_BOARD_DIMENSIONS = (WII_BOARD_WIDTH, WII_BOARD_LENGTH)

        self.data_streams: Dict[str, Any] = {}
        self.events: list = []
        self.use_wii = use_wii
        self.kwargs = kwargs

    def load_xdf(self, xdf_file: str = None) -> str:
        """Load XDF file using shared loader"""
        if xdf_file is None:
            try:
                from PyQt5.QtWidgets import QFileDialog, QApplication
                app = QApplication([])
                xdf_file, _ = QFileDialog.getOpenFileName(
                    None, "Select XDF File", "", "XDF Files (*.xdf);;All Files (*)"
                )
                app.quit()
            except ImportError:
                raise ImportError("PyQt5 is required for file dialog. Install with: pip install PyQt5")
        
        # Use shared XDF loader - FIX: Use correct attribute name
        self.streams, self.header = self.loader.load_file(xdf_file)
        
        # Organize streams using shared classifier - FIX: Use correct attribute name
        self.organized_streams = self.classifier.organize_streams(self.streams)
        
        return xdf_file

    def _extract_events(self):
        """Extract events from marker streams using shared components"""
        self.events = []
        
        marker_streams = self.organized_streams['marker_streams']
        
        for stream_idx, stream in enumerate(marker_streams):
            stream_name = stream['info'].get('name', [f'Stream_{stream_idx}'])[0]
            
            if 'time_stamps' in stream and 'time_series' in stream:
                timestamps = stream['time_stamps']
                markers = stream['time_series']
                
                for i, marker in enumerate(markers):
                    if len(marker) > 0 and marker[0]:
                        self.events.append({
                            'onset': timestamps[i],
                            'duration': 0.0,
                            'trial_type': str(marker[0]),
                            'source': stream_name
                        })
        
        # Sort by onset time
        self.events.sort(key=lambda x: x['onset'])
        logger.info(f"Extracted {len(self.events)} events")

    def _calculate_wii_cop(self, force_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate COP for Wii Balance Board (enhanced from original)"""
        if force_data.shape[1] < 4:
            return {'raw_data': force_data}
        
        # Force sensors: [TR, TL, BR, BL]
        TR, TL, BR, BL = force_data[:, 0], force_data[:, 1], force_data[:, 2], force_data[:, 3]
        total_force = TR + TL + BR + BL
        
        # Board dimensions in cm
        X, Y = self.WII_BOARD_DIMENSIONS
        
        # Calculate COP with improved numerical stability
        COPx = np.zeros_like(total_force)
        COPy = np.zeros_like(total_force)
        
        # Use a small threshold to avoid division by very small numbers
        valid_force = total_force > 1e-6  # Threshold for numerical stability
        
        if np.any(valid_force):
            COPx[valid_force] = (X/2) * ((TR[valid_force] + BR[valid_force]) - 
                                        (TL[valid_force] + BL[valid_force])) / total_force[valid_force]
            COPy[valid_force] = (Y/2) * ((TR[valid_force] + TL[valid_force]) - 
                                        (BR[valid_force] + BL[valid_force])) / total_force[valid_force]
        
        return {
            'COP_x': COPx,
            'COP_y': COPy,
            'force_total': total_force,
            'force_TR': TR,
            'force_TL': TL,
            'force_BR': BR,
            'force_BL': BL
        }

    def process_data(self, save_output: bool = True, output_dir: str = None) -> Dict[str, Any]:
        """Main processing pipeline using shared components"""
        if self.streams is None:
            raise ValueError("No XDF data loaded. Call load_xdf() first.")
        
        # Extract events
        self._extract_events()
        
        # Find overlap window using shared TimeUtils
        data_streams = self.organized_streams['data_streams']
        start_time, end_time = self.time_utils.find_overlap_window(data_streams)
        
        # Process each data stream in the overlap window
        processed_data = {}
        stream_metadata = {}
        
        for stream in data_streams:
            stream_name = stream['info'].get('name', ['Unknown'])[0]
            stream_type = self.classifier.classify_stream(stream)
            
            # Extract data in time window using shared TimeUtils
            timestamps = np.array(stream['time_stamps'])
            data = np.array(stream['time_series'])
            
            windowed_data, windowed_timestamps = self.time_utils.extract_time_window(
                data, timestamps, start_time, end_time
            )
            
            if len(windowed_data) == 0:
                continue
            
            # Process based on stream type
            if stream_type == 'wii' and self.use_wii:
                processed_stream_data = self._calculate_wii_cop(windowed_data)
            else:
                processed_stream_data = {'raw_data': windowed_data}
            
            # Store processed data
            processed_data[stream_type] = processed_stream_data
            processed_data[f'{stream_type}_timestamps'] = windowed_timestamps
            
            # Store metadata using shared ChannelParser
            channel_labels = self.parser.get_channel_labels(stream)
            stream_metadata[stream_type] = {
                'name': stream_name,
                'channel_count': int(stream['info'].get('channel_count', ['0'])[0]),
                'nominal_srate': float(stream['info'].get('nominal_srate', ['0'])[0]),
                'channel_labels': channel_labels,
                'samples': len(windowed_data)
            }
        
        # Compile results
        results = {
            'data': processed_data,
            'metadata': stream_metadata,
            'events': self.events,
            'time_window': (start_time, end_time),
            'processing_info': {
                'data_streams_processed': len(processed_data),
                'events_found': len(self.events),
                'duration': end_time - start_time
            }
        }
        
        logger.info(f"Processed {len(processed_data)} data streams with {len(self.events)} events")
        return results

    def export_to_bids(self, results: Dict[str, Any], output_path: str):
        """Export processed data to BIDS format using shared exporter"""
        # Use shared BIDS exporter - FIX: Use correct attribute name
        exported_files = self.exporter.export_to_bids(results, output_path)
        logger.info(f"BIDS export completed: {len(exported_files)} files created")
        return exported_files

    def preprocess_xdf(self, xdf_file: str = None, output_dir: str = None) -> Dict[str, Any]:
        """Complete preprocessing pipeline with shared components"""
        # Load data
        if xdf_file:
            xdf_file = self.load_xdf(xdf_file)
        elif self.streams is None:
            xdf_file = self.load_xdf()
        
        # Process data
        results = self.process_data()
        
        # Export if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename_base = os.path.splitext(os.path.basename(xdf_file))[0]
            output_path = os.path.join(output_dir, filename_base)
            self.export_to_bids(results, output_path)
            logger.info(f"Exported data to: {output_dir}")
        
        return results


# Convenience function
def process_xdf_file(xdf_file: str, output_dir: str = None, **kwargs) -> Dict[str, Any]:
    """Simple function to process an XDF file using shared components"""
    processor = XDFProcessor(**kwargs)
    return processor.preprocess_xdf(xdf_file, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    processor = XDFProcessor()
    results = processor.preprocess_xdf(output_dir=SAVE_FOLDER)
    
    print("Processing completed!")
    print(f"Processed {results['processing_info']['data_streams_processed']} streams")
    print(f"Found {results['processing_info']['events_found']} events")


