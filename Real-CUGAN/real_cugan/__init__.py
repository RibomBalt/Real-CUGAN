"""Real-CUGAN: 动漫图像/视频超分（2x/3x/4x）。"""

from __future__ import annotations

from .upcunet_v3 import RealWaifuUpScaler, UpCunet2x, UpCunet3x, UpCunet4x

__all__ = [
    "RealWaifuUpScaler",
    "UpCunet2x",
    "UpCunet3x",
    "UpCunet4x",
    "VideoRealWaifuUpScaler",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """按需导入视频模块，避免仅用图像推理时也要加载 moviepy。"""
    if name == "VideoRealWaifuUpScaler":
        from .inference_video import VideoRealWaifuUpScaler

        return VideoRealWaifuUpScaler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
