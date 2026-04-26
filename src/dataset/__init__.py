"""
src/dataset — dataset classes for PhysGTO experiments.

Public API:
    AeroGtoDataset      (3-D standard, dataset_fast.py)
    AeroGtoDataset2D    (2-D, dataset_2d.py)
    CutAeroGtoDataset   (spatially cropped, dataset_cut_fast.py)
    ShapeGroupedSampler (按 ds_shape 分桶 sampler，供图模型批量训练用)
"""

from src.dataset.dataset_fast import AeroGtoDataset, ShapeGroupedSampler
from src.dataset.dataset_cut_fast import CutAeroGtoDataset

__all__ = ["AeroGtoDataset", "CutAeroGtoDataset", "ShapeGroupedSampler"]
