"""图像读写：走 numpy + cv2.imdecode/imencode，避免 Windows 下中文路径读不出图。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

#: 参与批量推理的图像扩展名
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def imread_rgb(path: str | Path) -> np.ndarray:
    """读取图像并返回 HWC 的 RGB uint8 数组。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise OSError(f"无法读取图像（文件为空）：{path}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"无法解码图像（格式不支持或文件损坏）：{path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def imwrite_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """把 HWC 的 RGB uint8 数组写入文件。"""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", rgb[:, :, ::-1])
    if not ok:
        raise OSError(f"图像编码失败：{path}")
    buf.tofile(str(path))


def list_images(input_path: str | Path) -> list[Path]:
    """把输入路径展开为待处理图像列表（文件或目录，目录只取一层）。"""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def unique_path(directory: Path, stem: str, suffix: str = ".png") -> Path:
    """在目录下生成不冲突的文件路径，重名时追加 _1、_2 …"""
    candidate = directory / f"{stem}{suffix}"
    index = 0
    while candidate.exists():
        index += 1
        candidate = directory / f"{stem}_{index}{suffix}"
    return candidate


def format_exts(exts: Iterable[str]) -> str:
    return "、".join(exts)
