import importlib.util
import os
import sys
import time
from enum import Enum
from pathlib import Path

import torch
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from config.configuration import AllConfig, RefConfiguration

from DATT.learning.utils.feedforward_feature_extractor import FeedforwardFeaturesExtractor

from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.refs import TrajectoryRef
from DATT.schema import save_policy_manifest


class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        import tqdm

        self._bar = tqdm.tqdm(
            total=total_timesteps,
            desc="train",
            unit="steps",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        self._last_n = 0

    def _on_step(self):
        current = self.num_timesteps
        if current - self._last_n >= 100:
            self._bar.n = current
            self._bar.update(current - self._last_n)
            self._last_n = current
        return True

    def _on_training_end(self):
        self._bar.close()


def find_default_name_num(dir, prefix):
    seen_nums = set()
    for name in os.listdir(dir):
        if name.startswith(f"{prefix}_"):
            try:
                num = int(name[len(prefix) + 1 :])
            except ValueError:
                pass
            else:
                seen_nums.add(num)
    num = 0
    while num in seen_nums:
        num += 1
    return f"{prefix}_{num}"


def train(args):
    task: DroneTask = DroneTask(args.task)
    algo: RLAlgo = RLAlgo(args.algo)
    config_filename = args.config
    n_envs = args.n_envs

    # resolve policy name
    policy_name = args.name
    if policy_name is None:
        policy_name = f"{task.value}_{algo.value}_{int(time.time())}"

    config: AllConfig = import_config(config_filename)

    # resolve trajectory reference
    ref_str = args.ref
    if ref_str is None:
        ref_str = config.ref_config.default_ref
    ref = TrajectoryRef.get_by_value(ref_str)

    log_dir = DEFAULT_LOG_DIR / f"{policy_name}_logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if not (CONFIG_DIR / config_filename).exists():
        raise FileNotFoundError(f"{config_filename} is not a valid config file")

    env_kwargs = {
        "config": config,
        "save_data": False,
        "data_file": None,
    }

    if task.is_trajectory():
        env_kwargs["ref"] = ref
        seed = args.seed
        if seed is not None:
            env_kwargs["seed"] = seed
        else:
            env_kwargs["seed"] = np.random.randint(0, 100000)

    env_class = task.env()

    if issubclass(env_class, VecEnv):
        env = VecMonitor(env_class(n_envs))
    else:
        env = make_vec_env(env_class, n_envs=n_envs, env_kwargs=env_kwargs)

    algo_class = algo.algo_class()

    if not (SAVED_POLICY_DIR / f"{policy_name}.zip").exists():
        features_extractor_kwargs = {}
        if task.is_trajectory():
            if config.policy_config.conv_extractor:
                features_extractor_class = FeedforwardFeaturesExtractor
            else:
                features_extractor_class = FlattenExtractor

            features_extractor_kwargs["extra_state_features"] = 0
            extra_dims = task.env()(config=config).extra_dims
            features_extractor_kwargs["extra_state_features"] += extra_dims
            if task == DroneTask.TRAJFBFF or task == DroneTask.TRAJFBFF_VEL:
                features_extractor_kwargs["extra_state_features"] += 3
            elif task == DroneTask.TRAJFBFF_YAW:
                features_extractor_kwargs["extra_state_features"] += 4

            if task == DroneTask.TRAJFBFF and not config.policy_config.fb_term:
                features_extractor_kwargs["extra_state_features"] -= 3

            if task == DroneTask.TRAJFBFF_VEL:
                features_extractor_kwargs["dims"] = 6
            elif task == DroneTask.TRAJFBFF_YAW:
                features_extractor_kwargs["dims"] = 4
        else:
            features_extractor_class = FlattenExtractor

        print(f"Using feature extractor: {features_extractor_class}")
        net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
        if issubclass(algo_class, OffPolicyAlgorithm):
            net_arch = [64, 64, 64]

        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=net_arch,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )

        kwargs = {}
        if issubclass(algo_class, OffPolicyAlgorithm):
            kwargs["train_freq"] = (5000, "step")

        policy_network_type = "MlpPolicy"
        print(f"Using policy network type: {policy_network_type}")

        policy: BaseAlgorithm = algo_class(
            policy_network_type,
            env,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda:0",
            verbose=1,
            **kwargs,
        )
    else:
        policy: BaseAlgorithm = algo_class.load(SAVED_POLICY_DIR / f"{policy_name}.zip", env)
        policy.verbose = 1
        print("CONTINUING TRAINING!")

    progress_callback = TQDMProgressCallback(total_timesteps=args.timesteps)
    checkpoint_callback = CheckpointCallback(
        save_freq=250000, save_path=SAVED_POLICY_DIR, name_prefix=policy_name
    )
    cb = CallbackList([progress_callback, checkpoint_callback])
    policy.learn(total_timesteps=args.timesteps, callback=cb)
    policy.save(SAVED_POLICY_DIR / policy_name)
    save_policy_manifest(
        policy_name=policy_name,
        task=task.value,
        algo=algo.value,
        config=config,
        total_steps=args.timesteps,
        config_file=config_filename,
    )
    print(f"Saved policy manifest: {SAVED_POLICY_DIR / f'{policy_name}.policy.json'}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-t", "--task", default="trajectory_fbff")
    p.add_argument("-a", "--algo", default="ppo")
    p.add_argument("-c", "--config", default="DATT_config.py")
    p.add_argument("--ref", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-ts", "--timesteps", type=int, default=25_000_000)
    p.add_argument("--n-envs", type=int, default=10)
    train(p.parse_args())
