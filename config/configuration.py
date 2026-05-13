"""Backward-compatible re-exports from DATT.schema.

All config types are now Pydantic models defined in DATT.schema.
This module exists so existing ``from config.configuration import *``
statements in config/*.py files continue to work unchanged.
"""

from DATT.schema import (  # noqa: F401
    AdaptationConfiguration,
    AllConfig,
    ConfigValue,
    DroneConfiguration,
    EnvCondition,
    InitializationConfiguration,
    PolicyConfiguration,
    RefConfiguration,
    Sampler,
    SimConfiguration,
    TrainingConfiguration,
    WindConfiguration,
)
