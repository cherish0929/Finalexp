"""
src/contrast — contrast / baseline model architectures for PhysGTO experiments.

Public API:
    MGN              (mgn_model.py)           — MeshGraphNets
    GNOT             (gnot_model.py)          — General Neural Operator Transformer
    Transolver       (transolver_model.py)    — Physics-Attention Transformer
    GraphViTModel    (baseline_models.py)     — Graph + ViT hybrid
    FNO3DModel       (baseline_models.py)     — 3D Fourier Neural Operator
    UNet3DModel      (baseline_models.py)     — 3D U-Net
    MeltPoolResNet   (lpbf_baseline_models.py)— LPBF 3D ResNet surrogate
    ConvLSTMModel    (lpbf_baseline_models.py)— LPBF 3D ConvGRU surrogate
    ResNet3DModel    (lpbf_baseline_models.py)— LPBF volumetric ResNet

Author: contrast experiment extension
"""

from src.contrast.mgn_model import Model as MGN
from src.contrast.gnot_model import Model as GNOT
from src.contrast.transolver_model import Model as Transolver
from src.contrast.baseline_models import GraphViTModel, FNO3DModel, UNet3DModel
from src.contrast.lpbf_baseline_models import MeltPoolResNet, ConvLSTMModel, ResNet3DModel

__all__ = [
    "MGN", "GNOT", "Transolver",
    "GraphViTModel", "FNO3DModel", "UNet3DModel",
    "MeltPoolResNet", "ConvLSTMModel", "ResNet3DModel",
]
