from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShatterPattern:
    # An array with values from 0 to num_pieces-1 indicating piece assignments
    region_map: np.ndarray
    num_pieces: int
    width: int
    height: int


@dataclass(frozen=True)
class BoundingBox:
    min_row: int
    min_col: int
    max_row: int
    max_col: int


@dataclass(frozen=True)
class ShatterPiece:
    data: np.ndarray
    bbox: BoundingBox
    piece_id: int
