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
__all__ = ["XDFProcessor"]

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
        logging.error("ERROR: pyxdf requires importlib.metadata which is not available in Python < 3.8")
        logging.info("SOLUTION: Please install the backport package:")
        logging.info("  pip install importlib-metadata")
        logging.info("\nOr upgrade to Python 3.8+")
        logging.info("\nAlternatively, if you have pyxdf source code, you can modify:")
        logging.info("  pyxdf/__init__.py line 6:")
        logging.info("  Change: from importlib.metadata import PackageNotFoundError, version")
        logging.info("  To:     from importlib_metadata import PackageNotFoundError, version")
    else:
        logging.error(f"ERROR: Failed to import pyxdf: {error_msg}")
        logging.info("SOLUTION: Install pyxdf with: pip install pyxdf")

    # Create a dummy pyxdf module to prevent further import errors
    class DummyPyXDF:
        @staticmethod
        def load_xdf(*args, **kwargs):
            raise ImportError("pyxdf is not properly installed. See error message above.")
    
    import sys
    sys.modules['pyxdf'] = DummyPyXDF()
    pyxdf = DummyPyXDF()

from xdf2bids.utils import parse_event_string

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # console output
logger.addHandler(logging.StreamHandler())


SAVE_FOLDER = 'data/preprocessed/debug'
WII_BOARD_WIDTH = 43.3  # cm
WII_BOARD_LENGTH = 23.8  # cm
WII_CHANNELS = {"tl": "Weight_TopLeft", "tr": "Weight_TopRight", "bl": "Weight_BottomLeft", "br": "Weight_BottomRight"}

START_FROM_TRIAL = 0
TRIAL_END_PATTERN = 'TRIAL_END'
TRIAL_META_PATTERNS = ['trial_type', 'trial_name']

META_TABLE_ORDER =["onset", "duration", "trial_name", "trial_type", "has_perturbations", "has_movement"]
EVENT_TABLE_ORDER = ["onset", "duration", "label"]

DUPLICATE_ONSET_DECIMALS = 3
IGNORE_FOR_OVERLAP = ['online_cop']

WARNING_THRESHOLD = 0.3  # seconds

STREAM_PATTERNS = {
    'wii': 
        {'name': ['wii'], 
         'type': ['mocap'], 
         'matching':'pattern'},
        
    'kinect': 
        {'name': ['kinect', "kinectv2"], 
         'type': ['mocap', 'position'], 
         'matching':'pattern'},
        
    'eye_tracker': 
        {'name': ['neurolab_neon gaze', 'EyeLink'], 
         'type': ['gaze'], 
         'matching':'pattern'},
        
    'eye_tracker_events': 
        {'name': ['neurolab_neon events'], 
         'type': ['event'], 
         'matching':'pattern'},

    'eeg':
        {'name': ['liveamp', 'actichamp],
         'type': ['eeg', 'biosignal'],
         'matching':'pattern'},
    
    'online_cop':
        {'name': ['cop'],
            'type': ['cop'],
            'matching':'pattern'},
        
    'stimulus': 
        {'name': ['vr_bodysway', 'stimulus', 'stim'], 
         'type': ['timeseries'], 
         'matching':'pattern'},
        
    'marker': 
        {'name': ['marker', 'event', 'trigger'], 
         'type': ['marker', 'event'], 
         'matching':'pattern'},
        
    'meta': 
        {'name': ['trial_meta'], 
         'type': ['marker', 'meta'], 
         'matching':'pattern'},
}

class XDFProcessor:
    """Simplified XDF processor focused on loading and BIDS export"""

    def __init__(self, **kwargs):
        """Initialize with basic configuration."""

        # Basic stream identification patterns
        self.stream_patterns = STREAM_PATTERNS.copy()

        if 'stream_patterns' in kwargs:
            self.stream_patterns.update(kwargs['stream_patterns'])

        # Ensure patterns are lowercase for consistency
        self.stream_patterns = {k: {**v, 'name': [p.lower() for p in v['name']]} for k, v in self.stream_patterns.items()}

        # Data storage
        self.streams = None
        self.header = None
        self.data_streams = []
        self.marker_streams = []
        self.events = []
        self.meta = []
        self.global_t0 = None  # Store global time offset for external access

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
        
        logger.debug(f"Loading XDF file: {xdf_file}")
        
        try:
            self.streams, self.header = pyxdf.load_xdf(xdf_file)
        except ImportError as e:
            raise ImportError(f"Failed to load XDF file due to pyxdf import error: {e}")
        
        logger.debug(f"Loaded {len(self.streams)} streams")
        
        self._organize_streams()
        return xdf_file

    def _organize_streams(self):
        """Organize streams into data and marker streams"""
        self.data_streams = []
        self.marker_streams = []
        self.meta_streams = []
        
        for stream in self.streams:
                
            if self._is_meta_stream(stream):
                self.meta_streams.append(stream)
            
            elif self._is_marker_stream(stream):
                self.marker_streams.append(stream)
            # sort out empty streams
            elif ('time_stamps' not in stream or len(stream['time_stamps']) == 0):
                logger.warning(f"Stream '{stream['info'].get('name', ['Unnamed'])[0]}' is empty, skipping.")
            else:
                self.data_streams.append(stream)

        logger.debug(f"Found {len(self.data_streams)} data streams: {[self.data_streams[i]['info']['name'] for i in range(len(self.data_streams)) ]}, {len(self.marker_streams)} marker streams, {len(self.meta_streams)} meta streams")

    def _is_marker_stream(self, stream: Dict[str, Any]) -> bool:
        """Simple check if stream is a marker stream"""
        stream_type = stream['info'].get('type', [''])[0].lower()
        stream_name = stream['info'].get('name', [''])[0].lower()
        
        # Check by type or name patterns
        if any(indicator in stream_type for indicator in self.stream_patterns['marker']['type']) and \
            any(indicator in stream_name for indicator in self.stream_patterns['marker']['name']):
            return True
        else:
            return False

    def _is_meta_stream(self, stream: Dict[str, Any]) -> bool:
        """Check if stream is a metadata stream"""
        stream_type = stream['info'].get('type', [''])[0].lower()
        stream_name = stream['info'].get('name', [''])[0].lower()

        return any(indicator in stream_type for indicator in self.stream_patterns['meta']['type']) and \
               any(indicator in stream_name for indicator in self.stream_patterns['meta']['name'])

    def _classify_stream(self, stream: Dict[str, Any]) -> str:
        """Classify stream type based on name/type patterns"""
        stream_name = stream['info'].get('name', [''])[0].lower()
        stream_type_str = stream['info'].get('type', [''])[0].lower()

        for stream_type, patterns in self.stream_patterns.items():
            if patterns['matching'] == 'exact':
                if any(pattern == stream_name for pattern in patterns['name']) and any(pattern == stream_type_str for pattern in patterns['type']):
                    return stream_type
            elif patterns['matching'] == 'pattern':
                if any(pattern in stream_name for pattern in patterns['name']) and any(pattern in stream_type_str for pattern in patterns['type']):
                    return stream_type
        else:
            logger.warning(f"Stream '{stream_name}' does not match any known patterns, classifying as 'data'")
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
                        event_dict = {
                            'onset': timestamps[i],
                            'duration': 0.0,
                            'event_type': str(marker[0]),
                            'source': stream_name
                        }
                        event_dict.update(parse_event_string(event_dict['event_type']))
                        self.events.append(event_dict)
        # Deduplicate and sort using DataFrame, then convert back to dicts
        if self.events:
            events_df = pd.DataFrame(self.events)
            events_df['onset_rounded'] = events_df['onset'].round(DUPLICATE_ONSET_DECIMALS)
            events_df.drop_duplicates(subset=['label', 'onset_rounded'], inplace=True)
            events_df.drop(columns='onset_rounded', inplace=True)
            desired_order = EVENT_TABLE_ORDER
            remaining = [col for col in events_df.columns if col not in desired_order]
            events_df = events_df[desired_order + remaining]
            events_df.sort_values(by='onset', inplace=True)
            self.events = events_df.to_dict(orient='records')
        logger.debug(f"Extracted {len(self.events)} events")

    def _extract_meta(self):
        """Extract metadata from meta streams"""

        meta_records = []
        for idx, stream in enumerate(self.meta_streams):
            stream_name = stream['info'].get('name', [f'MetaStream_{idx}'])[0]
            if 'time_stamps' in stream and 'time_series' in stream:
                timestamps = stream['time_stamps']
                meta_data = stream['time_series']
                for i, data in enumerate(meta_data):
                    if len(data) > 0:
                        parsed = parse_event_string(data[0])
                        parsed['onset'] = timestamps[i]
                        parsed['source'] = stream_name
                        meta_records.append(parsed)

        if not meta_records:
            logger.warning("No meta events found in the streams")
            self.meta = []
            return

        meta_df = pd.DataFrame(meta_records)

        # Remove duplicates based on onset time (rounded)
        meta_df['onset_rounded'] = meta_df['onset'].round(DUPLICATE_ONSET_DECIMALS)
        # Only deduplicate if 'onset_rounded' and at least one other column exist
        dedup_cols = ['onset_rounded']
        if 'trial_name' in meta_df.columns:
            dedup_cols.append('trial_name')
        elif 'data' in meta_df.columns:
            dedup_cols.append('data')
        meta_df.drop_duplicates(subset=dedup_cols, inplace=True)
        meta_df.drop(columns='onset_rounded', inplace=True)

        # Remove 'data' column if it exists (no longer needed after parsing)
        if 'data' in meta_df.columns:
            meta_df.drop(columns='data', inplace=True)

        # Reorder columns: META_TABLE_ORDER first, then the rest
        final_order = [col for col in META_TABLE_ORDER if col in meta_df.columns] + \
                    [col for col in meta_df.columns if col not in META_TABLE_ORDER]
        meta_df = meta_df[final_order]

        # Sort by onset time
        meta_df.sort_values(by='onset', inplace=True)

        # Store as list of dicts for downstream compatibility
        self.meta = meta_df.to_dict(orient='records')

        logger.debug(f"Extracted {len(self.meta)} meta events")

    def _extract_trials_from_events(self, event_type = "trial_meta"):
        """Extract trial information and keep all events"""
        
        # Keep all events as-is for transparency
        # Just sort them by onset time
        events = pd.DataFrame(self.events)
        events = events.loc[events['event_type']==event_type]

        # Check if we have any events after filtering
        if events.empty:
            logger.warning(f"No events found with event_type '{event_type}'. Setting trials to empty list.")
            self.trials = []
            return

        # remove empty columns
        nan_value = float("NaN")
        events.replace("", nan_value, inplace=True)
        events.dropna(how='all', axis=1, inplace=True)
        events.sort_values(by='onset', inplace=True)

        # Correct trial durations if needed (should not be longer than time to next trial)
        if 'duration' in events.columns:
            for i in range(len(events)-1):
                current_onset = events.iloc[i]['onset']
                next_onset = events.iloc[i+1]['onset']
                current_duration = events.iloc[i]['duration']
                if current_onset + current_duration > next_onset:
                    corrected_duration = next_onset - current_onset
                    if np.abs(corrected_duration - current_duration) > WARNING_THRESHOLD: 
                        logger.warning(f"Correcting trial {i} duration from {current_duration} to {corrected_duration}")
                    events.at[events.index[i], 'duration'] = corrected_duration

        else:
            logger.warning("No 'duration' column found in trial events, skipping duration correction.")
        self.trials = events.to_dict(orient='records')



    def _extract_perturbations_from_events(self) -> List[Dict[str, Any]]:
        """Extract perturbation events from the event list"""
        perturbation_starts = []
        perturbation_ends = []
        
        for event in self.events:
            if 'perturbation_start' in event['event_type'].lower():
                perturbation_starts.append({
                    'onset': event['onset'],
                    'duration': 0.0,  # Duration will be calculated later
                    'event_type': event['event_type'],
                    'source': event['source']
                })
            elif 'perturbation_end' in event['event_type'].lower():
                perturbation_ends.append({
                    'onset': event['onset'],
                    'duration': 0.0,  # Duration will be calculated later
                    'event_type': event['event_type'],
                    'source': event['source']
                })

        # Sort perturbations by onset time
        perturbation_starts.sort(key=lambda x: x['onset'])
        perturbation_ends.sort(key=lambda x: x['onset'])

        # Remove duplicates: if onset times are very close, keep only the first occurrence
        def deduplicate_on_onset(events, atol=1e-5):
            deduped = []
            for ev in events:
                onset = ev['onset']
                if not any(np.isclose(onset, prev['onset'], atol=atol) for prev in deduped):
                    deduped.append(ev)
            return deduped

        perturbation_starts = deduplicate_on_onset(perturbation_starts)
        perturbation_ends = deduplicate_on_onset(perturbation_ends)

        try:
            assert len(perturbation_starts) == len(perturbation_ends), "Mismatched perturbation start/end counts"
            
            for start, end in zip(perturbation_starts, perturbation_ends):
                start['duration'] = end['onset'] - start['onset']
                start['event_type'] = 'perturbation'
                start['source'] = start['source']

            self.perturbations = perturbation_starts
            logger.debug(f"Extracted {len(self.perturbations)} perturbation events")

        except AssertionError as e:
            logger.warning(f"Mismatched perturbation start/end counts: {e}")
            if len(perturbation_starts) - len(perturbation_ends) == 1:
                logger.warning("Possible missing perturbation end event, using last start as end")
                
                for start, end in zip(perturbation_starts, perturbation_ends + [perturbation_starts[-1]]):
                    start['duration'] = end['onset'] - start['onset']
                    start['event_type'] = 'perturbation'
                    start['source'] = start['source']
                self.perturbations = perturbation_starts
            else:
                logger.error(f"Unable to resolve perturbation events due to mismatched counts: {e}")
                logger.debug("Number of starts: {}, Number of ends: {}".format(
                    len(perturbation_starts), len(perturbation_ends)))
                self.perturbations = []

    def _get_channel_labels(self, stream: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, str]]]:
        """Extract channel labels and all available metadata
        
        Returns:
            Tuple of (channel_labels, channel_metadata)
            where channel_metadata is a list of dicts containing all available channel info
        """
        try:
            ch_count = int(stream['info'].get('channel_count', ['0'])[0])
            channel_labels = []
            channel_metadata = []
            
            # Try to get from description
            if 'desc' in stream['info'] and stream['info']['desc']:
                desc = stream['info']['desc'][0]
                
                # Handle nested structure - try both paths
                channels_list = None
                
                # Path 1: desc -> channels -> channel (original nested structure)
                if hasattr(desc, 'get') and 'channels' in desc:
                    channels_info = desc['channels']
                    if isinstance(channels_info, list) and len(channels_info) > 0:
                        channels_dict = channels_info[0]
                        if hasattr(channels_dict, 'get') and 'channel' in channels_dict:
                            channels_list = channels_dict['channel']
                
                # Path 2: desc -> channel (flatter structure, e.g., eye tracker)
                if channels_list is None and hasattr(desc, 'get') and 'channel' in desc:
                    channels_list = desc['channel']
                
                # Extract all available metadata from channel list
                if channels_list:
                    for ch_idx, ch_info in enumerate(channels_list):
                        if hasattr(ch_info, 'get'):
                            # Extract label
                            label = ch_info.get('label', [''])[0] if 'label' in ch_info else ''
                            
                            # If label is empty, generate a default one
                            if not label:
                                label = f"Channel_{ch_idx+1}"
                            
                            # Dynamically extract all other metadata fields
                            metadata = {'label': label}
                            for key in ch_info.keys():
                                if key != 'label':  # label already extracted
                                    value = ch_info.get(key, [''])
                                    # Handle both list and non-list values
                                    if isinstance(value, list) and len(value) > 0:
                                        metadata[key] = value[0]
                                    elif not isinstance(value, list):
                                        metadata[key] = value
                                    else:
                                        metadata[key] = ''
                            
                            # Make label unique by appending eye/index if needed
                            # Check if this label already exists
                            if label in channel_labels:
                                # Try to make it unique with eye info
                                eye_info = metadata.get('eye', '')
                                if eye_info and eye_info != 'both':
                                    unique_label = f"{label}_{eye_info}"
                                else:
                                    # Count how many times this label appears
                                    count = channel_labels.count(label) + sum(1 for lbl in channel_labels if lbl.startswith(f"{label}_"))
                                    unique_label = f"{label}_{count}"
                                label = unique_label
                                metadata['label'] = unique_label
                            
                            channel_labels.append(label)
                            channel_metadata.append(metadata)
            
            # Fallback to numbered channels if we don't have enough
            while len(channel_labels) < ch_count:
                label = f"Channel_{len(channel_labels)+1}"
                channel_labels.append(label)
                channel_metadata.append({'label': label})
            
            # Fix potential collision with LSL timestamp column
            fixed_labels = []
            for i, label in enumerate(channel_labels[:ch_count]):
                if label.lower() == 'time':
                    fixed_label = 'trial_time'
                    logger.warning(f"Renamed data channel '{label}' to '{fixed_label}' to avoid collision with LSL timestamps")
                    fixed_labels.append(fixed_label)
                    channel_metadata[i]['label'] = fixed_label
                else:
                    fixed_labels.append(label)
            
            logger.debug(f"Extracted {len(fixed_labels)} channel labels from stream with {ch_count} channels")
            return fixed_labels, channel_metadata[:ch_count]
            
        except Exception as e:
            logger.warning(f"Error extracting channel labels: {e}")
            ch_count = int(stream['info'].get('channel_count', ['1'])[0])
            labels = [f"Channel_{i+1}" for i in range(ch_count)]
            metadata = [{'label': label} for label in labels]
            return labels, metadata

    def _find_overlap_window(self) -> Tuple[float, float]:
        """Find time window where all data streams overlap"""
        if not self.data_streams:
            return 0.0, 1.0
        
        start_times = []
        end_times = []
        
        for stream in self.data_streams:
            if stream['info'].get('name', [''])[0] in IGNORE_FOR_OVERLAP:
                continue
            timestamps = stream['time_stamps']
            if len(timestamps) > 0:
                start_times.append(timestamps[0])
                end_times.append(timestamps[-1])
                logger.debug(f"Stream '{stream['info'].get('name', ['Unnamed'])[0]}' time range: {timestamps[0]:.3f} to {timestamps[-1]:.3f} seconds")
        
        if not start_times:
            return 0.0, 1.0

        overlap_start = max(start_times)  # Latest start
        overlap_end = min(end_times)      # Earliest end

        # Log Events and Meta time ranges
        first_event, last_event = (min([e['onset'] for e in self.events]), max([e['onset'] for e in self.events])) if self.events else (None, None)
        first_trial, last_trial = (min([t['onset'] for t in self.trials]), max([t['onset'] + t['duration'] for t in self.trials])) if hasattr(self, 'trials') and self.trials else (None, None)
        first_meta, last_meta = (min([m['onset'] for m in self.meta]), max([m['onset'] for m in self.meta])) if self.meta else (None, None)

        logger.debug(f"Events start {first_event- overlap_start} to end {last_event - overlap_start} seconds relative to overlap start") if first_event is not None else None
        logger.debug(f"Trials start {first_trial - overlap_start} to end {last_trial - overlap_start} seconds relative to overlap start") if first_trial is not None else None
        logger.debug(f"Meta start {first_meta - overlap_start} to end {last_meta - overlap_start} seconds relative to overlap start") if first_meta is not None else None

        # Find experiment start time from events
        experiment_start = float(self.trials[START_FROM_TRIAL]['onset']) if hasattr(self, 'trials') and self.trials else None
        if experiment_start is not None and experiment_start > overlap_start:
            logger.debug(f"Adjusting overlap start time to experiment start time (trial {START_FROM_TRIAL}): {experiment_start:.3f} seconds")
            overlap_start = experiment_start
            return overlap_start, overlap_end
        elif overlap_start <= overlap_end:
            logger.debug(f"Overlap window: {overlap_start:.3f} to {overlap_end:.3f} seconds ({overlap_end-overlap_start:.3f}s duration)")
            return overlap_start, overlap_end
        else:
            logger.warning("No overlap found, using full time range")
            return min(start_times), max(end_times)

    def process_data(self, save_output: bool = True, output_dir: str = None, trial_event_type: str = "trial_meta") -> Dict[str, Any]:
        """Main processing pipeline"""
        if self.streams is None:
            raise ValueError("No XDF data loaded. Call load_xdf() first.")
        
        # Extract events
        self._extract_events()
        self._extract_meta()
        self._extract_trials_from_events(trial_event_type)
        self._extract_perturbations_from_events()
        
        # Find overlap window
        start_time, end_time = self._find_overlap_window()
        
        # Store global time offset for external access
        self.global_t0 = start_time
        self.global_t1 = end_time
        
        # Process each data stream in the overlap window
        processed_data = {}
        stream_metadata = {}
        
        for stream in self.data_streams:
            stream_name = stream['info'].get('name', ['Unknown'])[0]
            stream_type = self._classify_stream(stream)
            
            # Extract data in time window
            timestamps = stream['time_stamps']
            data = stream['time_series']
            channel_labels, channel_metadata = self._get_channel_labels(stream)
            
            # Filter to overlap window
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            windowed_timestamps = timestamps[mask]
            windowed_data = data[mask]
            
            if len(windowed_data) == 0:
                continue
            
            # Process based on stream type
            processed_stream_data = {'raw_data': windowed_data}
            
            # Store processed data
            processed_data[stream_type] = processed_stream_data
            processed_data[f'{stream_type}_timestamps'] = windowed_timestamps
            
            # Store metadata
            stream_metadata[stream_type] = {
                'name': stream_name,
                'channel_count': int(stream['info'].get('channel_count', ['0'])[0]),
                'nominal_srate': float(stream['info'].get('nominal_srate', ['0'])[0]),
                'effective_srate': float(stream['info'].get('effective_srate', ['0'])),
                'type': stream['info'].get('type', [''])[0],
                'channel_labels': channel_labels,
                'channel_metadata': channel_metadata,  # Store full channel metadata
                'samples': len(windowed_data)
            }
        
        # Compile results
        results = {
            'data': processed_data,
            'metadata': stream_metadata,
            'events': self.events,  # All raw events
            'meta': self.meta,  # All raw meta events
            'trials': getattr(self, 'trials', []),  # Clean trial table
            'perturbations': getattr(self, 'perturbations', []),  # Clean perturbation events
            'time_window': (start_time, end_time),
            'global_t0': self.global_t0,  # Global time offset for external use
            'processing_info': {
                'data_streams_processed': len(processed_data),
                'events_found': len(self.events),
                'trials_found': len(getattr(self, 'trials', [])),
                'duration': end_time - start_time
            }
        }
        
        logger.info(f"Processed {len(processed_data)} data streams with {len(self.events)} events")
        return results

    def convert_to_relative_time(self, absolute_timestamps: np.ndarray) -> np.ndarray:
        """Convert absolute LSL timestamps to relative timestamps using stored global_t0"""
        if self.global_t0 is None:
            raise ValueError("No global_t0 available. Run process_data() first.")
        return absolute_timestamps - self.global_t0
    
    def convert_to_absolute_time(self, relative_timestamps: np.ndarray) -> np.ndarray:
        """Convert relative timestamps back to absolute LSL timestamps using stored global_t0"""
        if self.global_t0 is None:
            raise ValueError("No global_t0 available. Run process_data() first.")
        return relative_timestamps + self.global_t0

    def apply_relative_time_to_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience function to convert processed results to relative time format.
        
        This creates a copy of the results with all timestamps converted to relative time
        (starting from 0) while preserving the original data.
        
        Args:
            results: Results dictionary from process_data()
            
        Returns:
            New results dictionary with relative timestamps
        """
        import copy
        
        # Create a deep copy to avoid modifying original data
        relative_results = copy.deepcopy(results)
        
        global_t0 = results['time_window'][0]
        
        # Convert stream timestamps to relative time
        for stream_type in relative_results['data']:
            if stream_type.endswith('_timestamps'):
                timestamps = relative_results['data'][stream_type]
                relative_results['data'][stream_type] = timestamps - global_t0
        
        # Convert event timestamps to relative time
        for event in relative_results['events']:
            event['onset'] = event['onset'] - global_t0
            
        # Convert meta event timestamps to relative time
        for meta_event in relative_results['meta']:
            meta_event['onset'] = meta_event['onset'] - global_t0
            
        # Convert trial timestamps to relative time
        for trial in relative_results['trials']:
            trial['onset'] = trial['onset'] - global_t0
        
        # Convert perturbation timestamps to relative time
        for perturbation in relative_results['perturbations']:
            perturbation['onset'] = perturbation['onset'] - global_t0
        
        # Mark as relative time
        relative_results['use_relative_time'] = True
        
        return relative_results

    def export_to_bids(self, results: Dict[str, Any], output_path: str):
        """Export processed data to BIDS format"""
        base_path = os.path.splitext(output_path)[0]
        global_t0 = results['time_window'][0]
        use_relative_time = results.get('use_relative_time', True)  # Default to relative time
        
        # Export data streams
        for stream_type in results['data']:
            if stream_type.endswith('_timestamps'):
                continue
                
            if stream_type in results['data'] and f'{stream_type}_timestamps' in results['data']:
                stream_data = results['data'][stream_type]
                timestamps = results['data'][f'{stream_type}_timestamps']
                metadata = results['metadata'].get(stream_type, {})
                
                # Timestamps are already processed according to use_relative_time flag
                if use_relative_time:
                    time_description = "Time relative to recording start"
                    start_time = float(timestamps[0])
                else:
                    time_description = "Absolute LSL timestamps"
                    start_time = float(timestamps[0])
                
                # Check if timestamps are monotonically increasing
                if not np.all(np.diff(timestamps) >= 0):
                    logger.warning(f"Timestamps for {stream_type} are not monotonically increasing. Adjusting...")
                    logger.debug(f"There is a risk of misalignment in the data due to non-monotonic timestamps.")
                    logger.debug(f"There is a jump in {len(np.where(np.diff(timestamps) < 0)[0 ])} places.")
                else:
                    logger.debug(f"Timestamps for {stream_type} are monotonically increasing.")
                
                # Prepare DataFrame
                df_data = {'time': timestamps}

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
                
                # Get actual channel count from data, not from stream info
                actual_channel_count = len(df.columns) - 1  # Subtract 1 for 'time' column
                channel_metadata_list = metadata.get('channel_metadata', [])
                
                sidecar = {
                    "SamplingFrequency": metadata.get('nominal_srate', 'n/a'),
                    "StartTime": start_time,
                    "Columns": list(df.columns),
                    "StreamType": stream_type,
                    "StreamName": metadata.get('name', 'Unknown'),
                    "ChannelCount": actual_channel_count,
                    "ChannelMetadata": channel_metadata_list[:actual_channel_count] if channel_metadata_list else None,
                    "Description": f"Data from {stream_type} stream",
                    "TimingInfo": {
                        "use_relative_time": use_relative_time,
                        "time_description": time_description,
                        "global_t0": float(global_t0),
                        "time_window": results['time_window']
                    }
                }
                
                # Remove None values
                sidecar = {k: v for k, v in sidecar.items() if v is not None}
                
                with open(json_path, 'w') as f:
                    json.dump(sidecar, f, indent=2)
                
                logger.debug(f"Exported {stream_type}: {len(df)} samples ({'relative' if use_relative_time else 'absolute'} time)")
        
        # Export events
        if results['events']:
            df_events = pd.DataFrame(results['events'])
            
            # Events are already processed according to use_relative_time flag
            if use_relative_time:
                onset_description = "Event onset time relative to recording start"
            else:
                onset_description = "Absolute LSL event onset time"

            if 'event_type' in df_events.columns and 'label' in df_events.columns:
                df_events.drop(columns='event_type', inplace=True)  # Remove 'event_type' column if exists

            # Save events TSV
            events_tsv = f"{base_path}_events.tsv"
            df_events.to_csv(events_tsv, sep='\t', index=False, float_format='%.6f')
            
            # Save events JSON
            events_json = f"{base_path}_events.json"
            events_sidecar = {
                "onset": {"Description": onset_description, "Units": "seconds"},
                "duration": {"Description": "Event duration in seconds", "Units": "seconds"},
                "event_type/label": {"Description": "Type of event or marker"},
                "source": {"Description": "Source stream name"},
                "timing_info": {
                    "use_relative_time": use_relative_time,
                    "global_t0": float(global_t0) if use_relative_time else None
                }
            }
            
            with open(events_json, 'w') as f:
                json.dump(events_sidecar, f, indent=2)
            
            logger.debug(f"Exported {len(df_events)} events ({'relative' if use_relative_time else 'absolute'} time)")

        # Export clean trials table
        if results['trials']:
            df_trials = pd.DataFrame(results['trials'])
            
            # Trials are already processed according to use_relative_time flag
            if use_relative_time:
                trial_onset_description = "Trial start time relative to recording start"
            else:
                trial_onset_description = "Absolute LSL trial start time"
            
            trials_tsv = f"{base_path}_trials.tsv"
            df_trials.to_csv(trials_tsv, sep='\t', index=False, float_format='%.6f')
            
            trials_json = f"{base_path}_trials.json"
            trials_sidecar = {
                "trial_number": {"Description": "Trial number (0-indexed)"},
                "onset": {"Description": trial_onset_description, "Units": "seconds"},
                "duration": {"Description": "Trial duration in seconds", "Units": "seconds"},
                "trial_type": {"Description": "Type of trial"},
                "timing_info": {
                    "use_relative_time": use_relative_time,
                    "global_t0": float(global_t0) if use_relative_time else None
                }
            }
            
            with open(trials_json, 'w') as f:
                json.dump(trials_sidecar, f, indent=2)
            
            logger.debug(f"Exported {len(df_trials)} clean trials ({'relative' if use_relative_time else 'absolute'} time)")
        # Export perturbations
        if results['perturbations']:
            df_perturbations = pd.DataFrame(results['perturbations'])
            perturbations_tsv = f"{base_path}_perturbations.tsv"
            df_perturbations.to_csv(perturbations_tsv, sep='\t', index=False, float_format='%.6f')

            perturbations_json = f"{base_path}_perturbations.json"
            perturbations_sidecar = {
                "onset": {"Description": "Perturbation onset time", "Units": "seconds"},
                "duration": {"Description": "Perturbation duration in seconds", "Units": "seconds"},
                "perturbation_type": {"Description": "Type of perturbation"},
                "timing_info": {
                    "use_relative_time": use_relative_time,
                    "global_t0": float(global_t0) if use_relative_time else None
                }
            }

            with open(perturbations_json, 'w') as f:
                json.dump(perturbations_sidecar, f, indent=2)

            logger.debug(f"Exported {len(df_perturbations)} perturbations ({'relative' if use_relative_time else 'absolute'} time)")
        
        if results['meta']:
            df_meta = pd.DataFrame(results['meta'])
            if 'data' in df_meta.columns:
                df_meta.drop(columns='data', inplace=True)  # Remove 'data' column if exists
            meta_tsv = f"{base_path}_meta.tsv"
            df_meta.to_csv(meta_tsv, sep='\t', index=False, float_format='%.6f')

            meta_json = f"{base_path}_meta.json"
            meta_sidecar = {
                "onset": {"Description": "Metadata event onset time", "Units": "seconds"},
                "data": {"Description": "Metadata content"},
                "source": {"Description": "Source stream name"},
                "timing_info": {
                    "use_relative_time": use_relative_time,
                    "global_t0": float(global_t0) if use_relative_time else None
                }
            }

            with open(meta_json, 'w') as f:
                json.dump(meta_sidecar, f, indent=2)

            logger.debug(f"Exported {len(df_meta)} metadata events ({'relative' if use_relative_time else 'absolute'} time)")
        # Save global metadata
        global_metadata = {
            "global_t0": float(global_t0) if use_relative_time else None,
            "use_relative_time": use_relative_time,
            "time_window": results['time_window'],
            "processing_info": results['processing_info']
        }
        global_json = f"{base_path}_global.json"
        with open(global_json, 'w') as f:
            json.dump(global_metadata, f, indent=2)

    def preprocess_xdf(self, xdf_file: str = None, output_dir: str = None, use_relative_time: bool = True, trial_event_type: str = "trial_meta") -> Dict[str, Any]:
        """Complete preprocessing pipeline
        
        Args:
            xdf_file: Path to XDF file to process
            output_dir: Directory to export BIDS files to
            use_relative_time: If True, timestamps start from 0. If False, use absolute LSL timestamps
            trial_event_type: Event type to use for trial extraction (default: "trial_meta")
        """
        # Load data
        if xdf_file:
            xdf_file = self.load_xdf(xdf_file)
        elif self.streams is None:
            xdf_file = self.load_xdf()
        
        # Process data (always preserves absolute timestamps)
        results = self.process_data(trial_event_type=trial_event_type)
        
        # Apply relative time conversion if requested
        if use_relative_time:
            results = self.apply_relative_time_to_results(results)
        else:
            results['use_relative_time'] = False
        
        # Export if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename_base = os.path.splitext(os.path.basename(xdf_file))[0]
            output_path = os.path.join(output_dir, filename_base)
            self.export_to_bids(results, output_path)
            logger.debug(f"Exported data to: {output_dir}")
        
        return results


# Convenience function
def process_xdf_file(xdf_file: str, output_dir: str = None, use_relative_time: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Simple function to process an XDF file
    
    Args:
        xdf_file: Path to XDF file
        output_dir: Output directory
        use_relative_time: If True, time starts at 0. If False, uses absolute LSL timestamps.
        **kwargs: Additional arguments passed to XDFProcessor
    """
    processor = XDFProcessor(**kwargs)
    return processor.preprocess_xdf(xdf_file, output_dir, use_relative_time=use_relative_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    processor = XDFProcessor()
    results = processor.preprocess_xdf(output_dir=SAVE_FOLDER)
    
    logging.info("Processing completed!")
    logging.info(f"Processed {results['processing_info']['data_streams_processed']} streams")
    logging.info(f"Found {results['processing_info']['events_found']} events")
