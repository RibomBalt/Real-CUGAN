"""权重文件与项目路径解析。"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR_NAME = "weights_v3"

#: 可用的降噪档位（与 weights_v3/ 中的文件名对应）
DENOISE_PRESETS = ("no-denoise", "denoise1x", "denoise2x", "denoise3x", "conservative")
DEFAULT_DENOISE = "denoise3x"

_SCALE_RE = re.compile(r"up([234])x", re.IGNORECASE)


def weights_dir() -> Path:
    """返回权重目录：优先当前工作目录，其次项目根目录。"""
    for candidate in (Path.cwd() / WEIGHTS_DIR_NAME, PROJECT_ROOT / WEIGHTS_DIR_NAME):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"找不到权重目录 {WEIGHTS_DIR_NAME}/（已尝试 {Path.cwd()} 与 {PROJECT_ROOT}），"
        "请从 README 提供的网盘/GitHub Release 下载权重后放入该目录"
    )


def default_weight_path(scale: int, denoise: str = DEFAULT_DENOISE) -> Path:
    """按倍率 + 降噪档位推导默认权重路径。"""
    if denoise not in DENOISE_PRESETS:
        raise ValueError(
            f"未知的降噪档位 {denoise!r}，可选：{', '.join(DENOISE_PRESETS)}"
        )
    wdir = weights_dir()
    path = wdir / f"up{scale}x-latest-{denoise}.pth"
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in wdir.glob("up*x-*.pth"))) or "无"
        raise FileNotFoundError(f"缺少权重文件 {path}；该目录下可用权重：{available}")
    return path


def scale_from_weight_name(path: Path) -> int | None:
    """从权重文件名（如 up2x-latest-denoise3x.pth）中解析倍率，解析不出返回 None。"""
    match = _SCALE_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def resolve_weight_path(
    weight: str | Path | None,
    scale: int,
    denoise: str | None = None,
) -> Path:
    """解析命令行给出的权重路径；未给出时按倍率/降噪档位推导，并校验倍率一致。"""
    if weight is None:
        return default_weight_path(scale, denoise or DEFAULT_DENOISE)

    path = Path(weight).expanduser()
    if not path.is_file() and not path.is_absolute():
        # 相对路径先按当前目录找，再退回项目根目录（方便从任意目录调用 CLI）
        fallback = PROJECT_ROOT / path
        if fallback.is_file():
            path = fallback
    if not path.is_file():
        raise FileNotFoundError(f"找不到权重文件 {path}")
    weight_scale = scale_from_weight_name(path)
    if weight_scale is not None and weight_scale != scale:
        raise ValueError(
            f"权重 {path.name} 是 {weight_scale}x 模型，与 --scale {scale} 不一致"
        )
    return path
