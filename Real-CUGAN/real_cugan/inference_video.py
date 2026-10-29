"""Real-CUGAN 视频超分推理（多线程解码/推理/编码流水线）与 CLI。

与原版 `inference_video.py` 流程一致：主线程负责解码与按序写帧，
每张卡上开 `--nt` 个线程跑模型，帧通过队列在两者之间传递。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
from collections.abc import Sequence
from multiprocessing import Queue
from pathlib import Path
from random import uniform
from time import sleep
from time import time as ttime

import torch
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from .cli import add_config_arg, add_model_args, model_overrides, section_overrides
from .config import ConfigError, load_settings
from .paths import resolve_weight_path, scale_from_weight_name
from .upcunet_v3 import RealWaifuUpScaler


class UpScalerMT(threading.Thread):
    """推理线程：从 inp_q 取 (idx, frame)，超分后把 (idx, frame) 放进 res_q。"""

    def __init__(
        self, inp_q, res_q, device, model, p_sleep, nt, tile, cache_mode, alpha, errors
    ):
        threading.Thread.__init__(self)
        self.device = device
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.cache_mode = cache_mode
        self.nt = nt
        self.p_sleep = p_sleep
        self.tile = tile
        self.alpha = alpha
        self.errors = errors  # 与主线程共享的异常列表

    def inference(self, tmp):
        idx, np_frame = tmp
        with torch.no_grad():
            res = self.model(np_frame, self.tile, self.cache_mode, self.alpha)
        if self.nt > 1:
            sleep(uniform(self.p_sleep[0], self.p_sleep[1]))
        return idx, res

    def run(self):
        while True:
            tmp = self.inp_q.get()
            if tmp is None:  # 收到结束信号
                break
            try:
                self.res_q.put(self.inference(tmp))
            except Exception as exc:  # noqa: BLE001 - 线程内需兜住任何异常并上报主线程
                self.errors.append(exc)
                break


class VideoRealWaifuUpScaler:
    def __init__(
        self,
        nt,
        n_gpu,
        scale,
        weight_path,
        half,
        tile,
        cache_mode,
        alpha,
        p_sleep,
        decode_sleep,
        encode_params,
    ):
        self.nt = nt
        self.n_gpu = n_gpu  # 每块 GPU 开 nt 个线程
        self.scale = scale
        self.weight_path = str(weight_path)
        self.encode_params = encode_params
        self.decode_sleep = decode_sleep

        self.inp_q = Queue(self.nt * self.n_gpu * 2)  # 待超分帧缓存上限
        self.res_q = Queue(self.nt * self.n_gpu * 2)  # 超分结果缓存上限
        self.errors: list[BaseException] = []  # 推理线程上报的异常
        self.threads = []
        for i in range(self.n_gpu):
            device = f"cuda:{i}"
            # 每张卡只加载一份模型，卡上所有线程共享
            model = RealWaifuUpScaler(scale, self.weight_path, half, device)
            for _ in range(self.nt):
                upscaler = UpScalerMT(
                    self.inp_q,
                    self.res_q,
                    device,
                    model,
                    p_sleep,
                    self.nt,
                    tile,
                    cache_mode,
                    alpha,
                    self.errors,
                )
                upscaler.start()
                self.threads.append(upscaler)

    def __call__(self, inp_path, opt_path):
        t0 = ttime()
        objVideoreader = VideoFileClip(filename=str(inp_path))
        try:
            w, h = objVideoreader.w, objVideoreader.h
            with tempfile.TemporaryDirectory(prefix="real_cugan_audio_") as tmp_dir:
                audiofile = None
                if objVideoreader.audio is not None:
                    audiofile = os.path.join(tmp_dir, "audio.m4a")
                    objVideoreader.audio.write_audiofile(audiofile, codec="aac")
                writer = FFMPEG_VideoWriter(
                    str(opt_path),
                    (w * self.scale, h * self.scale),
                    objVideoreader.fps,
                    ffmpeg_params=self.encode_params,
                    audiofile=audiofile,
                )
                try:
                    self._process(objVideoreader, writer)
                finally:
                    self._shutdown()
                    writer.close()
        finally:
            objVideoreader.close()

        print(inp_path, "done,time cost:", ttime() - t0)

    def _process(self, objVideoreader, writer):
        """解码 -> 入队 -> 按序写帧，直到所有帧处理完。"""
        total_frame = objVideoreader.n_frames
        now_idx = 0
        idx2res = {}
        idx = -1
        for idx, frame in enumerate(objVideoreader.iter_frames()):
            if idx % 10 == 0:
                print(f"已解码 {idx}/{int(total_frame)} 帧")
            self.inp_q.put((idx, frame))
            # 否则解帧会一直抢主进程 CPU，不给其他线程留预处理/后处理空间
            sleep(self.decode_sleep)
            now_idx = self._drain(self.res_q, idx2res, writer, now_idx)
            self._raise_worker_errors()

        idx += 1
        while True:
            now_idx = self._drain(self.res_q, idx2res, writer, now_idx)
            self._raise_worker_errors()
            if self.inp_q.qsize() == 0 and self.res_q.qsize() == 0 and idx == now_idx:
                break
            sleep(0.02)

    def _shutdown(self):
        """通知所有推理线程退出（需在等待前调用，否则线程会一直阻塞在队列上）。"""
        for _ in range(self.nt * self.n_gpu):
            self.inp_q.put(None)
        for thread in self.threads:
            thread.join(timeout=5)

    def _raise_worker_errors(self):
        if self.errors:
            raise RuntimeError(
                f"帧推理失败（{len(self.errors)} 个线程报错）：{self.errors[0]}"
            )

    @staticmethod
    def _drain(res_q, idx2res, writer, now_idx):
        """把已完成的帧按 idx 顺序写入视频，返回下一个待写的 idx。"""
        while not res_q.empty():
            iidx, res = res_q.get()
            idx2res[iidx] = res
        while now_idx in idx2res:
            writer.write_frame(idx2res.pop(now_idx))
            now_idx += 1
        return now_idx


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="real-cugan-video",
        description="Real-CUGAN 视频超分：按 --n-gpu 分配显卡，每卡 --nt 线程推理，输出带原音轨的 mp4",
    )
    add_config_arg(parser)
    parser.add_argument(
        "-i", "--input", default=None, help="输入视频路径（默认取配置 video.input）"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出视频路径，mp4（默认取配置 video.output）",
    )
    add_model_args(parser)
    parser.add_argument("--nt", type=int, default=None, help="每张卡上的推理线程数")
    parser.add_argument("--n-gpu", type=int, default=None, help="使用的显卡数量")
    parser.add_argument(
        "--p-sleep",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="多线程推理时每帧的随机休眠区间，防止线程抢占",
    )
    parser.add_argument(
        "--decode-sleep",
        type=float,
        default=None,
        help="解帧后主线程休眠时间，避免解帧占满 CPU",
    )
    parser.add_argument(
        "--encode-params",
        nargs="+",
        default=None,
        metavar="ARG",
        help="传给 ffmpeg 的编码参数，例如 -crf 21 -preset medium",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    overrides = model_overrides(args)
    overrides.update(
        section_overrides(
            args,
            "video",
            (
                "input",
                "output",
                "nt",
                "n_gpu",
                "p_sleep",
                "decode_sleep",
                "encode_params",
            ),
        )
    )
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    try:
        settings = load_settings(args.config, _cli_overrides(args))
    except ConfigError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    video = settings.video

    try:
        weight_path = resolve_weight_path(
            settings.weights, settings.scale, settings.denoise
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    scale = scale_from_weight_name(weight_path) or settings.scale

    if video.input is None or video.output is None:
        print(
            "错误：缺少输入或输出视频路径。用 -i/-o 指定，或写进 config/config.yml 的 video.input / video.output",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "错误：视频模式需要 CUDA（按卡号自动分配 device）。"
            "请检查显卡驱动/torch 安装；纯 CPU 请用 real-cugan-image 逐帧处理",
            file=sys.stderr,
        )
        return 2
    available_gpus = torch.cuda.device_count()
    if video.n_gpu > available_gpus:
        print(
            f"错误：n_gpu {video.n_gpu} 超过可用显卡数 {available_gpus}",
            file=sys.stderr,
        )
        return 2

    inp_path = Path(video.input)
    if not inp_path.is_file():
        print(f"错误：输入视频不存在：{inp_path}", file=sys.stderr)
        return 2
    opt_path = Path(video.output)
    opt_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"权重: {weight_path}\n倍率: {scale}x  半精度: {settings.half}  tile: {settings.tile}  "
        f"cache_mode: {settings.cache_mode}  alpha: {settings.alpha:g}  "
        f"nt: {video.nt}  n_gpu: {video.n_gpu}"
    )
    upscaler = VideoRealWaifuUpScaler(
        video.nt,
        video.n_gpu,
        scale,
        weight_path,
        settings.half,
        settings.tile,
        settings.cache_mode,
        settings.alpha,
        tuple(video.p_sleep),
        video.decode_sleep,
        list(video.encode_params),
    )
    try:
        upscaler(inp_path, opt_path)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"完成：{opt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
