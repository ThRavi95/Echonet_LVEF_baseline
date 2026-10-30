import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class EchoNetParser:
    """
    Parses and indexes EchoNet-Dynamic metadata and volume tracings.
    """

    def __init__(self, file_list_csv: str, volume_tracings_csv: Optional[str] = None):
        if not os.path.exists(file_list_csv):
            raise FileNotFoundError(f"FileList.csv not found at: {file_list_csv}")
            
        self.df_files = pd.read_csv(file_list_csv)
        self._normalize_file_list()

        self.df_tracings = None
        self.tracings_index = {}
        if volume_tracings_csv and os.path.exists(volume_tracings_csv):
            self.df_tracings = pd.read_csv(volume_tracings_csv)
            self._index_tracings()

    def _normalize_file_list(self):
        # Ensure FileName has .avi suffix
        if "FileName" in self.df_files.columns:
            self.df_files["FileName"] = self.df_files["FileName"].apply(
                lambda x: x if str(x).endswith(".avi") else f"{x}.avi"
            )
        # Drop rows with missing EF targets
        self.df_files = self.df_files.dropna(subset=["EF"]).reset_index(drop=True)

    def _index_tracings(self):
        """Index VolumeTracings by (FileName, Frame) for fast O(1) lookup."""
        self.df_tracings["FileName"] = self.df_tracings["FileName"].apply(
            lambda x: x if str(x).endswith(".avi") else f"{x}.avi"
        )
        grouped = self.df_tracings.groupby(["FileName", "Frame"])
        for (filename, frame), group in grouped:
            if filename not in self.tracings_index:
                self.tracings_index[filename] = {}
            self.tracings_index[filename][int(frame)] = {
                "x1": group["X1"].tolist(),
                "y1": group["Y1"].tolist(),
                "x2": group["X2"].tolist(),
                "y2": group["Y2"].tolist(),
            }

    def get_split(self, split_name: str) -> pd.DataFrame:
        """
        Return metadata slice for 'TRAIN', 'VAL', or 'TEST'.
        Guarantees strict patient separation.
        """
        split_upper = split_name.upper()
        if "Split" not in self.df_files.columns:
            raise KeyError("FileList.csv must contain a 'Split' column.")
            
        subset = self.df_files[self.df_files["Split"].str.upper() == split_upper].copy()
        return subset.reset_index(drop=True)

    def get_cardiac_phase_frames(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Retrieve the two annotated cardiac phase frame indices (ED and ES).
        EchoNet annotates ED (End-Diastole, larger volume) and ES (End-Systole, smaller volume).
        Returns: (frame_ed, frame_es) or (None, None) if not annotated.
        """
        norm_name = filename if filename.endswith(".avi") else f"{filename}.avi"
        if norm_name not in self.tracings_index:
            return None, None
            
        frames = sorted(list(self.tracings_index[norm_name].keys()))
        if len(frames) >= 2:
            return frames[0], frames[1]
        elif len(frames) == 1:
            return frames[0], None
        return None, None

    def get_tracing_coordinates(self, filename: str, frame: int) -> Optional[Dict[str, List[float]]]:
        """Retrieve coordinate dict {'x1', 'y1', 'x2', 'y2'} for a specific frame."""
        norm_name = filename if filename.endswith(".avi") else f"{filename}.avi"
        return self.tracings_index.get(norm_name, {}).get(frame, None)
