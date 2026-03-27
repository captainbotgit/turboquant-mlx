"""TurboQuant-MLX: 3-bit KV cache compression for MLX on Apple Silicon.

References:
    - TurboQuant: arXiv:2504.19874
    - PolarQuant: arXiv:2502.02617
"""

from .codebook import get_gaussian_codebook, get_polar_codebook, pack_indices, unpack_indices, pack_signs, unpack_signs
from .patch import TurboQuantKVCache, patch_model, unpatch_model
from .polar_quant import PolarQuantConfig, PolarQuantLayer, PolarQuantState
from .qjl import QJLProjection, QJLState
from .turbo import TurboQuantCompressor, TurboQuantConfig, TurboQuantState

__version__ = "0.1.0"

__all__ = [
    "patch_model",
    "unpatch_model",
    "TurboQuantKVCache",
    "TurboQuantCompressor",
    "TurboQuantConfig",
    "TurboQuantState",
    "PolarQuantLayer",
    "PolarQuantConfig",
    "PolarQuantState",
    "QJLProjection",
    "QJLState",
    "get_gaussian_codebook",
    "get_polar_codebook",
]
