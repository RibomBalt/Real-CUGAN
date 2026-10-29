"""图像 / 视频两个 CLI 共享的参数与工具。

优先级：`config/config.yml` < 环境变量 / `.env` < 命令行参数。
因此这里所有参数的默认值都是 None，只有显式传入时才会覆盖配置。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .paths import DEFAULT_DENOISE, DENOISE_PRESETS, PROJECT_ROOT

CONFIG_HINT = (
    f"（默认读取 {PROJECT_ROOT / 'config' / 'config.yml'}，可被环境变量与命令行覆盖）"
)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=f"配置文件路径{CONFIG_HINT}",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """添加模型推理相关的通用参数（默认值来自配置文件）。"""
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        choices=(2, 3, 4),
        default=None,
        help="超分倍率 2/3/4",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="权重文件路径；不给则按 --scale/--denoise 从 weights_v3/ 推导",
    )
    parser.add_argument(
        "-n",
        "--denoise",
        choices=DENOISE_PRESETS,
        default=None,
        help=f"降噪档位，仅在未指定 --weights 时生效（默认 {DEFAULT_DENOISE}）",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=None,
        help="分块数：越大越省显存、越慢；0 为整图推理",
    )
    parser.add_argument(
        "-c",
        "--cache-mode",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="显存/速度权衡：0 正常、1 8bit 量化缓存、2 fast_rough 低显存、3 gap_sync 无损低显存",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=None,
        help="处理强度：(0.75, 1.3) 区间内调整，越大越模糊、越少 AI 痕迹",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="半精度推理，省显存（CPU 上自动关闭；用 --no-half 关闭）",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        help="推理设备，如 cuda:0 / cuda:1 / cpu（视频模式按 --n-gpu 自动分配）",
    )


def model_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """收集命令行中显式给出的模型参数，未给出的项不会进入结果。"""
    names = (
        "scale",
        "weights",
        "denoise",
        "tile",
        "cache_mode",
        "alpha",
        "half",
        "device",
    )
    return {
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    }


def section_overrides(
    args: argparse.Namespace, section: str, names: tuple[str, ...]
) -> dict:
    """把命令行的某个分组参数整理成嵌套覆盖项，例如 {"video": {"nt": 4}}。"""
    values = {
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    }
    return {section: values} if values else {}


def output_stem(
    stem: str,
    scale: int,
    tile: int,
    cache_mode: int,
    alpha: float,
    weight_path: Path,
) -> str:
    """沿用仓库原有命名：{原名}_{倍率}x_tile{tile}_cache{cache}_alpha{alpha}_{权重名}"""
    return (
        f"{stem}_{scale}x_tile{tile}_cache{cache_mode}"
        f"_alpha{alpha:g}_{Path(weight_path).stem}"
    )
