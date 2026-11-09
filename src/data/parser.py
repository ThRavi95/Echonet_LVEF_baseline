"""
EchoNet-Dynamic Metadata Parser
================================
Provides strictly split-aware access to FileList.csv and VolumeTracings.csv.

Cardiac phase assignment (ED / ES):
  EchoNet-Dynamic annotates exactly two frames per video — one at End-Diastole
  (maximum LV filling) and one at End-Systole (maximum LV contraction).  The
  relative order in the CSV is *not* guaranteed to equal chronological cardiac
  phase order.  We therefore compute the enclosed LV polygon area for each
  annotated frame and assign:
    • ed_frame  ← frame with the LARGER  area  (End-Diastole, peak volume)
    • es_frame  ← frame with the SMALLER area  (End-Systole,  peak ejection)
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.rasterizer import rasterize_lv_mask


class EchoNetParser:
    """
    Parses and indexes EchoNet-Dynamic FileList.csv and VolumeTracings.csv.

    Provides:
      • Strict split-based subset retrieval (TRAIN / VAL / TEST).
      • Area-based cardiac phase frame lookup (ED, ES) without chronological assumption.
      • Per-frame tracing coordinate retrieval.
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
        # tracings_index[filename][frame_int] = {x1, y1, x2, y2}
        self.tracings_index: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

        if volume_tracings_csv and os.path.exists(volume_tracings_csv):
            self.df_tracings = pd.read_csv(volume_tracings_csv)
            self._index_tracings()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_file_list(self) -> None:
        """Ensure FileName has .avi suffix and drop rows with missing EF."""
        if "FileName" in self.df_files.columns:
            self.df_files["FileName"] = self.df_files["FileName"].apply(
                lambda x: x if str(x).endswith(".avi") else f"{x}.avi"
            )
        self.df_files = self.df_files.dropna(subset=["EF"]).reset_index(drop=True)

    def _index_tracings(self) -> None:
        """Build a two-level dict for O(1) tracing lookup by (filename, frame)."""
        self.df_tracings["FileName"] = self.df_tracings["FileName"].apply(
            lambda x: x if str(x).endswith(".avi") else f"{x}.avi"
        )
        grouped = self.df_tracings.groupby(["FileName", "Frame"])
        for (filename, frame), grp in grouped:
            fname = str(filename)
            fidx = int(frame)
            self.tracings_index.setdefault(fname, {})[fidx] = {
                "x1": grp["X1"].tolist(),
                "y1": grp["Y1"].tolist(),
                "x2": grp["X2"].tolist(),
                "y2": grp["Y2"].tolist(),
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, split_name: str) -> pd.DataFrame:
        """
        Return the metadata slice for a named split.

        Args:
            split_name: One of 'TRAIN', 'VAL', 'TEST' (case-insensitive).

        Returns:
            Filtered DataFrame with the official Split column guarantee.
            Patient-level leakage is prevented by the upstream EchoNet-Dynamic
            split assignment in FileList.csv.

        Raises:
            KeyError: If the 'Split' column is absent (malformed FileList.csv).
        """
        if "Split" not in self.df_files.columns:
            raise KeyError(
                "FileList.csv must contain a 'Split' column "
                "with values TRAIN / VAL / TEST."
            )
        mask = self.df_files["Split"].str.upper() == split_name.upper()
        return self.df_files[mask].copy().reset_index(drop=True)

    def get_cardiac_phase_frames(
        self, filename: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Return the ED and ES frame indices for a video.

        The two annotated frames are disambiguated by computing the enclosed LV
        polygon area (number of non-zero pixels in the rasterised binary mask):

          ed_frame  ← frame with the LARGER  area  (End-Diastole)
          es_frame  ← frame with the SMALLER area  (End-Systole)

        Args:
            filename: Base name of the video (with or without .avi suffix).

        Returns:
            (ed_frame, es_frame) as integer frame indices, or (None, None) if
            the video has no tracing entries.
        """
        norm_name = filename if filename.endswith(".avi") else f"{filename}.avi"
        if norm_name not in self.tracings_index:
            return None, None

        frame_map = self.tracings_index[norm_name]
        frame_indices = sorted(frame_map.keys())

        if len(frame_indices) == 0:
            return None, None
        if len(frame_indices) == 1:
            return frame_indices[0], None

        # Compute enclosed area for every annotated frame
        areas: List[Tuple[int, int]] = []  # (frame_idx, pixel_area)
        for fidx in frame_indices:
            coords = frame_map[fidx]
            mask = rasterize_lv_mask(
                coords["x1"],
                coords["y1"],
                coords["x2"],
                coords["y2"],
                orig_shape=(112, 112),   # EchoNet-Dynamic canonical frame size
                target_shape=(112, 112),
            )
            areas.append((fidx, int(mask.sum())))

        # Sort by area descending: largest → ED, smallest → ES
        areas.sort(key=lambda t: t[1], reverse=True)
        ed_frame = areas[0][0]
        es_frame = areas[-1][0]

        return ed_frame, es_frame

    def get_tracing_coordinates(
        self, filename: str, frame: int
    ) -> Optional[Dict[str, List[float]]]:
        """
        Retrieve the Simpson's disk coordinate dict for a specific frame.

        Returns:
            {'x1': [...], 'y1': [...], 'x2': [...], 'y2': [...]} or None.
        """
        norm_name = filename if filename.endswith(".avi") else f"{filename}.avi"
        return self.tracings_index.get(norm_name, {}).get(int(frame), None)
