"""
src/dataset — dataset classes for PhysGTO experiments.

Public API:
    AeroGtoDataset      (3-D standard, dataset_fast.py)
    AeroGtoDataset2D    (2-D, dataset_2d.py)
    CutAeroGtoDataset   (spatially cropped, dataset_cut_fast.py)
    LPBFSlotDataset     (slot-based LPBF, dataset_lpbf_v3.py)
    LPBFLaserDataset    (slot-based LPBF + laser physics, dataset_lpbf.py)
"""

from src.dataset.dataset_fast import AeroGtoDataset
from src.dataset.dataset_2d import AeroGtoDataset2D
from src.dataset.dataset_cut_fast import CutAeroGtoDataset
from src.dataset.dataset_lpbf_v3 import LPBFSlotDataset
from src.dataset.dataset_lpbf import LPBFLaserDataset

__all__ = [
    "AeroGtoDataset",
    "AeroGtoDataset2D",
    "CutAeroGtoDataset",
    "LPBFSlotDataset",
    "LPBFLaserDataset",
]
