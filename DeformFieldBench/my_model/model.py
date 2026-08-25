# -*- coding: utf-8 -*-
"""Arch4 模型工厂（与 skills/模型搭建 的 dict 输出约定一致）。"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .arch4_model import Arch4VideoMAEPhysModel, build_arch4_model


def create_arch4_model(**kwargs: Any) -> Arch4VideoMAEPhysModel:
    return build_arch4_model(**kwargs)


def create_model(arch: str = "arch4", **kwargs: Any) -> nn.Module:
    a = str(arch).strip().lower()
    if a in ("arch4", "arch4_videomae_phys"):
        return create_arch4_model(**kwargs)
    raise ValueError(f"未知 arch={arch!r}，当前仅支持 arch4")
