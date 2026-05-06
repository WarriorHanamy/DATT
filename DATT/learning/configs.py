import os
import sys
from enum import Enum
from pathlib import Path
import importlib.util
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC


class RLAlgo(Enum):
    PPO = "ppo"
    A2C = "a2c"
    DDPG = "ddpg"
    SAC = "sac"
    TD3 = "td3"

    def algo_class(self):
        return {
            RLAlgo.PPO: PPO,
            RLAlgo.A2C: A2C,
            RLAlgo.DDPG: DDPG,
            RLAlgo.SAC: SAC,
            RLAlgo.TD3: TD3,
        }[RLAlgo(self._value_)]


thisdir = os.path.dirname(os.path.realpath(__file__))
ROOT = Path(thisdir).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "logs"
DEFAULT_DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)
SAVED_POLICY_DIR = ROOT / "checkpoints"
SAVED_POLICY_DIR.mkdir(exist_ok=True)


def import_config(config_filename):
    spec = importlib.util.spec_from_file_location(
        "datt_runtime_config", CONFIG_DIR / config_filename
    )

    config_module = importlib.util.module_from_spec(spec)
    sys.modules["datt_runtime_config"] = config_module
    spec.loader.exec_module(config_module)
    try:
        return config_module.config
    except AttributeError:
        raise ValueError(
            f"Config file {config_filename} must define a config object named `config`."
        )
