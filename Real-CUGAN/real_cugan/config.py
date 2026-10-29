"""配置加载：`config/config.yml` < 环境变量（含 `.env`）< 命令行参数。

- 结构定义与校验由 Pydantic V2 负责，字段写错、取值越界都会在启动时报错；
- 本机相关路径、显卡号等可以放到项目根目录 `.env`（模板见 `.env.example`），
  环境变量命名规则：`REAL_CUGAN_` + 字段名大写，嵌套字段用双下划线，
  例如 `REAL_CUGAN_VIDEO__INPUT` 对应 `video.input`；
- 命令行参数只覆盖显式给出的项，其余仍取配置文件 / 环境变量的值。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .paths import DENOISE_PRESETS, PROJECT_ROOT

ENV_PREFIX = "REAL_CUGAN_"
ENV_DELIMITER = "__"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_ENCODE_PARAMS = ["-crf", "21", "-preset", "medium"]


class ConfigError(Exception):
    """配置文件缺失、格式错误或校验失败。"""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageSection(_Section):
    input: Path = Path("inputs")
    output_dir: Path = Path("output")


class VideoSection(_Section):
    input: Path | None = None
    output: Path | None = None
    nt: int = Field(default=2, ge=1)
    n_gpu: int = Field(default=1, ge=1)
    p_sleep: tuple[float, float] = (0.005, 0.012)
    decode_sleep: float = Field(default=0.002, ge=0)
    encode_params: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ENCODE_PARAMS)
    )

    @field_validator("p_sleep")
    @classmethod
    def _check_p_sleep(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] < 0 or value[1] < 0:
            raise ValueError("p_sleep 不能为负数")
        if value[0] > value[1]:
            raise ValueError(f"p_sleep 需要 (最小值, 最大值)，当前 {value} 顺序反了")
        return value


class Settings(_Section):
    """推理参数。模型相关的顶层字段对图像/视频两个命令共用。"""

    # 用 int + 显式取值校验而不是 Literal，这样环境变量里的字符串 "3" 也能被正确解析
    scale: int = Field(default=2, ge=2, le=4)
    denoise: str = "denoise3x"
    weights: Path | None = None
    tile: int = Field(default=5, ge=0)
    cache_mode: int = Field(default=1, ge=0, le=3)
    alpha: float = Field(default=1.0, gt=0)
    half: bool = True
    device: str = "cuda:0"
    image: ImageSection = Field(default_factory=ImageSection)
    video: VideoSection = Field(default_factory=VideoSection)

    @field_validator("scale")
    @classmethod
    def _check_scale(cls, value: int) -> int:
        if value not in (2, 3, 4):
            raise ValueError(f"scale 只能是 2 / 3 / 4，当前 {value}")
        return value

    @field_validator("denoise")
    @classmethod
    def _check_denoise(cls, value: str) -> str:
        if value not in DENOISE_PRESETS:
            raise ValueError(
                f"denoise 只能是 {', '.join(DENOISE_PRESETS)}，当前 {value!r}"
            )
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并，override 中的标量覆盖 base 中的同名标量。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_layer() -> dict[str, Any]:
    """把 `REAL_CUGAN_*` 环境变量（含 `.env`）整理成嵌套字典。"""
    env: dict[str, str] = {
        key: value
        for key, value in dotenv_values(ENV_FILE).items()
        if value is not None
    }
    env.update(os.environ)  # 真实环境变量优先于 .env

    layer: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split(ENV_DELIMITER) if p]
        if not parts:
            continue
        node = layer
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(
                node, dict
            ):  # 例如同时设置了 REAL_CUGAN_VIDEO 和 REAL_CUGAN_VIDEO__NT
                raise ConfigError(f"环境变量 {key} 与其它 REAL_CUGAN_* 变量冲突")
        node[parts[-1]] = value
    return layer


def load_settings(
    config_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """按「默认值 < config.yml < 环境变量/.env < 命令行」的优先级合成配置。"""
    explicit_path = config_path is not None
    path = Path(config_path).expanduser() if explicit_path else DEFAULT_CONFIG_PATH

    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"配置文件不是合法的 YAML：{path}\n{exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"配置文件顶层必须是键值表：{path}")
        data = loaded
    elif explicit_path:
        raise ConfigError(f"配置文件不存在：{path}")

    data = _deep_merge(data, _env_layer())
    data = _deep_merge(
        data, {k: v for k, v in (cli_overrides or {}).items() if v is not None}
    )

    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"配置校验失败（来源：{path} + 环境变量 + 命令行）：\n{exc}"
        ) from exc
