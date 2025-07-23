""" xdf_processor.py - XDF Processor for BIDS Export of Motion and Eye Tracking Data 
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

__version__ = "0.1.0"

import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

try:
    import pyxdf
except (ImportError, ModuleNotFoundError) as e:
    error_msg = str(e)
    if "importlib.metadata" in error_msg:
        print("ERROR: pyxdf requires importlib.metadata which is not available in Python < 3.8")
        print("SOLUTION: Please install the backport package:")
        print("  pip install importlib-metadata")
        print("\nOr upgrade to Python 3.8+")
        print("\nAlternatively, if you have pyxdf source code, you can modify:")
        print("  pyxdf/__init__.py line 6:")
        print("  Change: from importlib.metadata import PackageNotFoundError, version")
        print("  To:     from importlib_metadata import PackageNotFoundError, version")
    else:
        print(f"ERROR: Failed to import pyxdf: {error_msg}")
        print("SOLUTION: Install pyxdf with: pip install pyxdf")
    
    # Create a dummy pyxdf module to prevent further import errors
    class DummyPyXDF:
        @staticmethod
        def load_xdf(*args, **kwargs):
            raise ImportError("pyxdf is not properly installed. See error message above.")
    
    import sys
    sys.modules['pyxdf'] = DummyPyXDF()
    pyxdf = DummyPyXDF()

logger = logging.getLogger(__name__)

SAVE_FOLDER = 'data/preprocessed'
WII_BOARD_WIDTH = 43.3  # cm
WII_BOARD_LENGTH = 23.8  # cm


class XDFProcessor:
    """Simplified XDF processor focused on loading and BIDS export"""

    def __init__(self, use_wii=True, **kwargs):
        """Initialize with basic configuration."""
        self.use_wii = use_wii
        
        # Device-specific Parameters
        self.WII_BOARD_DIMENSIONS = kwargs.get('WII_BOARD_DIMENSIONS', (WII_BOARD_WIDTH, WII_BOARD_LENGTH)) # Wii Balance Board dimensions in cm

        # Basic stream identification patterns
        self.stream_patterns = {
            'wii': ['wiiuse', 'wii'],
            'kinect': ['kinect', 'mocap'],
            'eye_tracker': ['pupil', 'eye'],
            'stimulus': ['vr_bodysway', 'stimulus', 'stim'],
            'marker': ['marker', 'event', 'trigger']
        }
        if 'stream_patterns' in kwargs:
            self.stream_patterns.update(kwargs['stream_patterns'])
        
        # Data storage
        self.streams = None
        self.header = None
        self.data_streams = []
        self.marker_streams = []
        self.events = []

    def load_xdf(self, xdf_file: str = None) -> str:
        """Load XDF file"""
        if xdf_file is None:
            try:
                from PyQt5.QtWidgets import QFileDialog, QApplication
                app = QApplication([])
                xdf_file, _ = QFileDialog.getOpenFileName(None, "Select XDF File", "", "XDF Files (*.xdf);;All Files (*)")
                app.quit()
            except ImportError:
                raise ImportError("PyQt5 is required for file dialog. Install with: pip install PyQt5")
            
        if not os.path.exists(xdf_file):
            raise FileNotFoundError(f"XDF file not found: {xdf_file}")
        
        logger.info(f"Loading XDF file: {xdf_file}")
        
        try:
            self.streams, self.header = pyxdf.load_xdf(xdf_file)
        except ImportError as e:
            raise ImportError(f"Failed to load XDF file due to pyxdf import error: {e}")
        
        logger.info(f"Loaded {len(self.streams)} streams")
        
        self._organize_streams()
        return xdf_file

    def _organize_streams(self):
        """Organize streams into data and marker streams"""
        self.data_streams = []
        self.marker_streams = []
        
        for stream in self.streams:
            if self._is_marker_stream(stream):
                self.marker_streams.append(stream)
            else:
                self.data_streams.append(stream)
        
        logger.info(f"Found {len(self.data_streams)} data streams and {len(self.marker_streams)} marker streams")

    def _is_marker_stream(self, stream: Dict[str, Any]) -> bool:
        """Simple check if stream is a marker stream"""
        stream_type = stream['info'].get('type', [''])[0].lower()
        stream_name = stream['info'].get('name', [''])[0].lower()
        
        # Check by type or name patterns
        if any(indicator in stream_type for indicator in self.stream_patterns['marker']):
            return True
        if any(indicator in stream_name for indicator in self.stream_patterns['marker']):
            return True
            
        # Check if data is string-based (typical for markers)
        time_series = stream.get('time_series', [])
        if isinstance(time_series, list) and len(time_series) > 0:
            first_sample = time_series[0]
            if isinstance(first_sample, (list, tuple)) and len(first_sample) > 0:
                if isinstance(first_sample[0], str):
                    return True
        
        return False

    def _classify_stream(self, stream: Dict[str, Any]) -> str:
        """Classify stream type based on name patterns"""
        stream_name = stream['info'].get('name', [''])[0].lower()
        
        for stream_type, patterns in self.stream_patterns.items():
            if any(pattern in stream_name for pattern in patterns):
                return stream_type
        
        return 'data'  # Default classification

    def _extract_events(self):
        """Extract events from marker streams"""
        self.events = []
        
        for stream_idx, stream in enumerate(self.marker_streams):
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

    def _get_channel_labels(self, stream: Dict[str, Any]) -> List[str]:
        """Extract channel labels with simplified parsing"""
        try:
            ch_count = int(stream['info'].get('channel_count', ['0'])[0])
            channel_labels = []
            
            # Try to get from description
            if 'desc' in stream['info'] and stream['info']['desc']:
                desc = stream['info']['desc'][0]
                
                # Handle nested structure
                if hasattr(desc, 'get') and 'channels' in desc:
                    channels_info = desc['channels']
                    if isinstance(channels_info, list) and len(channels_info) > 0:
                        channels_dict = channels_info[0]
                        if hasattr(channels_dict, 'get') and 'channel' in channels_dict:
                            for ch_info in channels_dict['channel']:
                                if hasattr(ch_info, 'get'):
                                    label = ch_info.get('label', [''])[0] if 'label' in ch_info else ''
                                    if label:
                                        channel_labels.append(label)
            
            # Fallback to numbered channels
            while len(channel_labels) < ch_count:
                channel_labels.append(f"Channel_{len(channel_labels)+1}")
            
            return channel_labels[:ch_count]
            
        except Exception as e:
            logger.warning(f"Error extracting channel labels: {e}")
            ch_count = int(stream['info'].get('channel_count', ['1'])[0])
            return [f"Channel_{i+1}" for i in range(ch_count)]

    def _calculate_wii_cop(self, force_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Simple COP calculation for Wii Balance Board"""
        if force_data.shape[1] < 4:
            return {'raw_data': force_data}
        
        # Force sensors: [TR, TL, BR, BL]
        TR, TL, BR, BL = force_data[:, 0], force_data[:, 1], force_data[:, 2], force_data[:, 3]
        total_force = TR + TL + BR + BL
        
        # Board dimensions in cm
        X, Y = self.WII_BOARD_DIMENSIONS
        
        # Calculate COP (avoid division by zero)
        COPx = np.zeros_like(total_force)
        COPy = np.zeros_like(total_force)
        
        valid_force = total_force > 0
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

    def _find_overlap_window(self) -> Tuple[float, float]:
        """Find time window where all data streams overlap"""
        if not self.data_streams:
            return 0.0, 1.0
        
        start_times = []
        end_times = []
        
        for stream in self.data_streams:
            timestamps = stream['time_stamps']
            if len(timestamps) > 0:
                start_times.append(timestamps[0])
                end_times.append(timestamps[-1])
        
        if not start_times:
            return 0.0, 1.0
        
        overlap_start = max(start_times)  # Latest start
        overlap_end = min(end_times)      # Earliest end
        
        if overlap_start <= overlap_end:
            logger.info(f"Overlap window: {overlap_start:.3f} to {overlap_end:.3f} seconds ({overlap_end-overlap_start:.3f}s duration)")
            return overlap_start, overlap_end
        else:
            logger.warning("No overlap found, using full time range")
            return min(start_times), max(end_times)

    def process_data(self, save_output: bool = True, output_dir: str = None) -> Dict[str, Any]:
        """Main processing pipeline"""
        if self.streams is None:
            raise ValueError("No XDF data loaded. Call load_xdf() first.")
        
        # Extract events
        self._extract_events()
        
        # Find overlap window
        start_time, end_time = self._find_overlap_window()
        
        # Process each data stream in the overlap window
        processed_data = {}
        stream_metadata = {}
        
        for stream in self.data_streams:
            stream_name = stream['info'].get('name', ['Unknown'])[0]
            stream_type = self._classify_stream(stream)
            
            # Extract data in time window
            timestamps = stream['time_stamps']
            data = stream['time_series']
            
            # Filter to overlap window
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            windowed_timestamps = timestamps[mask]
            windowed_data = data[mask]
            
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
            
            # Store metadata
            channel_labels = self._get_channel_labels(stream)
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
        """Export processed data to BIDS format"""
        base_path = os.path.splitext(output_path)[0]
        global_t0 = results['time_window'][0]
        
        # Export data streams
        for stream_type in results['data']:
            if stream_type.endswith('_timestamps'):
                continue
                
            if stream_type in results['data'] and f'{stream_type}_timestamps' in results['data']:
                stream_data = results['data'][stream_type]
                timestamps = results['data'][f'{stream_type}_timestamps']
                metadata = results['metadata'].get(stream_type, {})
                
                # Create relative timestamps
                relative_time = timestamps - global_t0
                
                # Prepare DataFrame
                df_data = {'time': relative_time}
                
                if stream_type == 'wii' and 'COP_x' in stream_data:
                    # Special handling for Wii data
                    df_data.update({
                        'COP_x': stream_data['COP_x'],
                        'COP_y': stream_data['COP_y'],
                        'force_total': stream_data['force_total'],
                        'force_TR': stream_data['force_TR'],
                        'force_TL': stream_data['force_TL'],
                        'force_BR': stream_data['force_BR'],
                        'force_BL': stream_data['force_BL']
                    })
                else:
                    # Standard data handling
                    raw_data = stream_data.get('raw_data', stream_data)
                    if isinstance(raw_data, np.ndarray) and len(raw_data.shape) > 1:
                        channel_labels = metadata.get('channel_labels', [])
                        for ch in range(raw_data.shape[1]):
                            if ch < len(channel_labels):
                                col_name = channel_labels[ch].replace(' ', '_')
                            else:
                                col_name = f'channel_{ch+1}'
                            df_data[col_name] = raw_data[:, ch]
                
                df = pd.DataFrame(df_data)
                
                # Save TSV file
                tsv_path = f"{base_path}_{stream_type}.tsv"
                df.to_csv(tsv_path, sep='\t', index=False, float_format='%.6f')
                
                # Save JSON sidecar
                json_path = f"{base_path}_{stream_type}.json"
                sidecar = {
                    "SamplingFrequency": metadata.get('nominal_srate', 'n/a'),
                    "StartTime": float(relative_time[0]),
                    "Columns": list(df.columns),
                    "StreamType": stream_type,
                    "StreamName": metadata.get('name', 'Unknown'),
                    "ChannelCount": metadata.get('channel_count', len(df.columns) - 1),
                    "Description": f"Data from {stream_type} stream",
                    "TimingInfo": {
                        "global_t0": float(global_t0),
                        "time_window": results['time_window']
                    }
                }
                
                with open(json_path, 'w') as f:
                    json.dump(sidecar, f, indent=2)
                
                logger.info(f"Exported {stream_type}: {len(df)} samples")
        
        # Export events
        if results['events']:
            df_events = pd.DataFrame(results['events'])
            
            # Adjust event times to be relative to global t0
            df_events['onset'] = df_events['onset'] - global_t0
            
            # Save events TSV
            events_tsv = f"{base_path}_events.tsv"
            df_events.to_csv(events_tsv, sep='\t', index=False, float_format='%.6f')
            
            # Save events JSON
            events_json = f"{base_path}_events.json"
            events_sidecar = {
                "onset": {"Description": "Event onset time in seconds", "Units": "seconds"},
                "duration": {"Description": "Event duration in seconds", "Units": "seconds"},
                "trial_type": {"Description": "Type of event or marker"},
                "source": {"Description": "Source stream name"}
            }
            
            with open(events_json, 'w') as f:
                json.dump(events_sidecar, f, indent=2)
            
            logger.info(f"Exported {len(df_events)} events")

    def preprocess_xdf(self, xdf_file: str = None, output_dir: str = None) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
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
    """Simple function to process an XDF file"""
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


