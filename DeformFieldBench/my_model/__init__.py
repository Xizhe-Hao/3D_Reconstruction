# -*- coding: utf-8 -*-
"""延迟导入重型依赖（如 torch），便于 ``python -m my_model.smoke_train_stages`` 等子模块先执行。"""

from typing import Any

__all__ = [
    "Arch4VideoMAEPhysModel",
    "Arch4LossConfig",
    "Arch4RegressionLoss",
    "DatasetArch4",
    "arch4_field_supervision_mse",
    "build_arch4_model",
    "create_model",
    "resolve_flat_dataset_root",
]


def __getattr__(name: str) -> Any:
    if name == "Arch4VideoMAEPhysModel":
        from .arch4_model import Arch4VideoMAEPhysModel

        return Arch4VideoMAEPhysModel
    if name == "build_arch4_model":
        from .arch4_model import build_arch4_model

        return build_arch4_model
    if name == "DatasetArch4":
        from .dataset import DatasetArch4

        return DatasetArch4
    if name == "resolve_flat_dataset_root":
        from .dataset import resolve_flat_dataset_root

        return resolve_flat_dataset_root
    if name == "Arch4LossConfig":
        from .losses import Arch4LossConfig

        return Arch4LossConfig
    if name == "Arch4RegressionLoss":
        from .losses import Arch4RegressionLoss

        return Arch4RegressionLoss
    if name == "arch4_field_supervision_mse":
        from .losses import arch4_field_supervision_mse

        return arch4_field_supervision_mse
    if name == "create_model":
        from .model import create_model

        return create_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
