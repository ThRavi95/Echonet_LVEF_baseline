"""
EchoNet-Dynamic Metadata Parser
================================
Provides strictly split-aware access to FileList.csv and VolumeTracings.csv.

Cardiac phase assignment (ED / ES):
  EchoNet-Dynamic annotates exactly two frames per video — one at End-Diastole
  (maximum LV filling) and one at End-Systole (maximum LV contraction).
    • ed_frame  ← frame with the LARGER  area  (End-Diastole, peak volume)
    • es_frame  ← frame with the SMALLER area  (End-Systole,  peak ejection)
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.rasterizer import rasterize_lv_mask


def _clean_stem(name: str) -> str:
    s = str(name).strip()
    return os.path.splitext(s)[0].strip()


class EchoNetParser:
    """
    Parses and indexes EchoNet-Dynamic FileList.csv and VolumeTracings.csv.
    """

    def __init__(
        self,
        file_list_csv: str,
        volume_tracings_csv: Optional[str] = None,
    ):
        if not os.path.exists(file_list_csv):
            raise FileNotFoundError(
                f"FileList.csv not found: {file_list_csv}\n"
                "Ensure the EchoNet-Dynamic dataset is correctly placed."
            )

        self.df_files = pd.read_csv(file_list_csv)
        self._normalize_file_list()

        self.df_tracings: Optional[pd.DataFrame] = None
        # tracings_index[norm_key][frame_int] = {x1, y1, x2, y2}
        self.tracings_index: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

        if volume_tracings_csv and os.path.exists(volume_tracings_csv):
            self.df_tracings = pd.read_csv(volume_tracings_csv)
            self._index_tracings()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_file_list(self) -> None:
        """Strip whitespaces, preserve original extensions, drop missing EF."""
        if "FileName" in self.df_files.columns:
            def _norm_name(val):
                s = str(val).strip()
                base, ext = os.path.splitext(s)
                if ext.lower() in (".avi", ".mp4", ".mov", ".npy"):
                    return s
                return f"{s}.avi"
            self.df_files["FileName"] = self.df_files["FileName"].apply(_norm_name)
        self.df_files = self.df_files.dropna(subset=["EF"]).reset_index(drop=True)

    def _index_tracings(self) -> None:
        """Build case-insensitive O(1) tracing lookup."""
        grouped = self.df_tracings.groupby(["FileName", "Frame"])
        for (filename, frame), grp in grouped:
            fname = str(filename).strip()
            fstem = _clean_stem(fname).lower()
            fname_lower = fname.lower()
            fidx = int(frame)
            coords = {
                "x1": grp["X1"].tolist(),
                "y1": grp["Y1"].tolist(),
                "x2": grp["X2"].tolist(),
                "y2": grp["Y2"].tolist(),
            }

            for key in (fname, fname_lower, fstem, f"{fstem}.avi", f"{fstem}.AVI"):
                self.tracings_index.setdefault(key, {})[fidx] = coords

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, split_name: str) -> pd.DataFrame:
        if "Split" not in self.df_files.columns:
            raise KeyError(
                "FileList.csv must contain a 'Split' column "
                "with values TRAIN / VAL / TEST."
            )
        mask = self.df_files["Split"].str.upper() == split_name.upper()
        return self.df_files[mask].copy().reset_index(drop=True)

    def _lookup_tracings(self, filename: str) -> Optional[Dict[int, Dict[str, List[float]]]]:
        s = str(filename).strip()
        stem = _clean_stem(s).lower()
        candidates = [s, s.lower(), stem, f"{stem}.avi", f"{stem}.AVI"]
        for c in candidates:
            if c in self.tracings_index:
                return self.tracings_index[c]
        return None

    def get_cardiac_phase_frames(
        self, filename: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Return the ED and ES frame indices for a video.
        """
        frame_map = self._lookup_tracings(filename)
        if not frame_map:
            return None, None

        frame_indices = sorted(frame_map.keys())
        if len(frame_indices) == 0:
            return None, None
        if len(frame_indices) == 1:
            return frame_indices[0], None

        areas: List[Tuple[int, int]] = []
        for fidx in frame_indices:
            coords = frame_map[fidx]
            mask = rasterize_lv_mask(
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"],
                orig_shape=(112, 112),
                target_shape=(112, 112),
            )
            areas.append((fidx, int(mask.sum())))

        areas.sort(key=lambda t: t[1], reverse=True)
        ed_frame = areas[0][0]
        es_frame = areas[-1][0]
        return ed_frame, es_frame

    def get_tracing_coordinates(
        self, filename: str, frame: int
    ) -> Optional[Dict[str, List[float]]]:
        frame_map = self._lookup_tracings(filename)
        if not frame_map:
            return None
        return frame_map.get(int(frame), None)
