"""图像 / 视频两个 CLI 共享的模型参数与输出命名。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .paths import DEFAULT_DENOISE, DENOISE_PRESETS


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """添加模型推理相关的通用参数。"""
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        choices=(2, 3, 4),
        default=2,
        help="超分倍率（默认 2）",
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
        default=DEFAULT_DENOISE,
        help=f"降噪档位，仅在未指定 --weights 时生效（默认 {DEFAULT_DENOISE}）",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=5,
        help="分块数：越大越省显存、越慢；0 为整图推理（默认 5）",
    )
    parser.add_argument(
        "-c",
        "--cache-mode",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help="显存/速度权衡：0 正常、1 8bit 量化缓存(默认)、2 fast_rough 低显存、3 gap_sync 无损低显存",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="处理强度：(0.75, 1.3) 区间内调整，越大越模糊、越少 AI 痕迹（默认 1.0）",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用半精度推理，省显存（默认开启，CPU 上会自动关闭；用 --no-half 关闭）",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="推理设备，如 cuda:0 / cuda:1 / cpu（默认 cuda:0；视频模式按 --n-gpu 自动分配）",
    )


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
