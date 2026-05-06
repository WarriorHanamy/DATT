import time
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from os.path import exists

from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.refs import TrajectoryRef

from DATT.quadsim.visualizer import Vis
from DATT.python_utils.plotu import subplot
from scipy.spatial.transform import Rotation as R
from config.configuration import AllConfig
from DATT.learning.utils.adaptation_network import AdaptationNetwork
from DATT.learning.base_env import BaseQuadsimEnv

from stable_baselines3.common.env_util import make_vec_env


def _add_e_dim(fbff_obs: np.ndarray, e: np.ndarray, base_dims=10):
    obs = np.concatenate((fbff_obs[:, :base_dims], e, fbff_obs[:, base_dims:]), axis=1)
    return obs


def _remove_e_dim(output_obs: np.ndarray, e_dims: int, base_dims=10, include_extra=False):
    if include_extra:
        obs = np.concatenate(
            [output_obs[:, :base_dims], output_obs[:, (base_dims + e_dims) :]], axis=1
        )
    else:
        obs = output_obs[:, :base_dims]
    return obs


def _rollout_adaptive_policy(
    rollout_len,
    adaptation_network,
    policy,
    evalenv,
    n_envs,
    time_horizon,
    base_dims,
    e_dims,
    device,
    vis,
    rate,
    progress=None,
):
    action_dims = 4

    history = torch.zeros((n_envs, base_dims + action_dims, time_horizon)).to(device)
    fbff_obs = _remove_e_dim(evalenv.reset(), e_dims, include_extra=True)
    all_states = []
    des_traj = []
    for i in range(rollout_len):
        e_pred = adaptation_network(history)

        input_obs = _add_e_dim(fbff_obs, e_pred.detach().cpu().numpy(), base_dims)

        actions, _states = policy.predict(input_obs, deterministic=True)

        obs, rewards, dones, info = evalenv.step(actions)

        e_gt = obs[:, base_dims : (base_dims + e_dims)]

        print(e_pred.detach().cpu().numpy(), "gt: ", e_gt)

        base_states = _remove_e_dim(obs, e_dims)
        adaptation_input = np.concatenate((base_states, actions), axis=1)

        adaptation_input = torch.from_numpy(adaptation_input).to(device).float()

        history = torch.cat(
            (torch.unsqueeze(adaptation_input, -1), history[:, :, :-1].clone()), dim=2
        )

        state = evalenv.get_attr("quadsim", 0)[0].rb.state()

        vis.set_state(state.pos.copy(), state.rot)
        if rate > 0:
            time.sleep(1.0 / rate)
        all_states.append(np.r_[state.pos, state.vel, obs[0, 6:10]])

        des_traj.append(evalenv.get_attr("ref", 0)[0].pos(evalenv.get_attr("t", 0)[0]))

        fbff_obs = _remove_e_dim(obs, e_dims, include_extra=True)

        if progress is not None:
            progress[0].update(task_id=progress[1], completed=i + 1)

    return np.array(all_states), np.array(des_traj)


def do_eval_rma(args):
    task: DroneTask = DroneTask(args.task)
    policy_name = args.name
    if policy_name is None:
        raise ValueError("--name is required (policy name to load)")
    algo: RLAlgo = RLAlgo(args.algo)
    config_filename = args.config
    adapt_name = args.adapt_name

    ref_str = args.ref
    if ref_str is None:
        config_pre: AllConfig = import_config(config_filename)
        ref_str = config_pre.ref_config.default_ref
    ref = TrajectoryRef.get_by_value(ref_str)

    seed = args.seed

    if not exists(SAVED_POLICY_DIR / f"{policy_name}.zip"):
        raise ValueError(f"policy not found: {policy_name}")
    if not exists(CONFIG_DIR / config_filename):
        raise FileNotFoundError(f"{config_filename} is not a valid config file")

    algo_class = algo.algo_class()

    config: AllConfig = import_config(config_filename)
    adapt_config = config.adapt_config
    env_params = adapt_config.include
    time_horizon = adapt_config.time_horizon

    dummy_env = BaseQuadsimEnv(config)
    e_dims = 0
    for param in env_params:
        _, dims, _, _ = param.get_attribute(dummy_env)
        e_dims += dims

    if seed is None:
        seed = np.random.randint(0, 100000)
        fixed_seed = False
    else:
        fixed_seed = True
    print("seed", seed)

    trainenv = task.env()(config=config)
    evalenv = make_vec_env(
        task.env(),
        n_envs=1,
        env_kwargs={
            "config": config,
            "ref": ref,
            "seed": seed,
            "fixed_seed": fixed_seed,
        },
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = algo_class.load(SAVED_POLICY_DIR / f"{policy_name}.zip")

    if adapt_name is None:
        adapt_name = f"{policy_name}_adapt"

    action_dims = 4
    if os.path.exists(SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}"):
        adaptation_network_state_dict = torch.load(
            SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}",
            map_location=torch.device("cpu"),
        )
    elif os.path.exists(SAVED_POLICY_DIR / f"{adapt_name}"):
        adaptation_network_state_dict = torch.load(
            SAVED_POLICY_DIR / f"{adapt_name}", map_location=torch.device("cpu")
        )
    else:
        raise ValueError(f"Invalid adaptation network name: {adapt_name}")

    adaptation_network = AdaptationNetwork(
        input_dims=trainenv.base_dims + action_dims, e_dims=e_dims
    )
    adaptation_network = adaptation_network.to(device)
    adaptation_network.load_state_dict(adaptation_network_state_dict)

    vis = Vis()
    while True:
        all_states, des_traj = _rollout_adaptive_policy(
            args.steps,
            adaptation_network,
            policy,
            evalenv,
            1,
            time_horizon,
            trainenv.base_dims,
            e_dims,
            device,
            vis,
            args.rate,
        )
        if args.viz:
            plt.figure()
            ax = plt.subplot(3, 1, 1)
            plt.plot(range(args.steps), all_states[:, 0])
            if des_traj.size > 0:
                plt.plot(range(args.steps), des_traj[:, 0])
            plt.subplot(3, 1, 2)
            plt.plot(range(args.steps), all_states[:, 1])
            if des_traj.size > 0:
                plt.plot(range(args.steps), des_traj[:, 1])
            plt.subplot(3, 1, 3)
            plt.plot(range(args.steps), all_states[:, 2])
            if des_traj.size > 0:
                plt.plot(range(args.steps), des_traj[:, 2])
            plt.suptitle("PPO (sim) des vs. actual pos")

            eulers = np.array(
                [R.from_quat(rot).as_euler("ZYX")[::-1] for rot in all_states[:, 6:10]]
            )
            subplot(range(args.steps), eulers, yname="Euler (rad)", title="ZYX Euler Angles")
            plt.show()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-t", "--task", default="trajectory_fbff")
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-an", "--adapt-name", default=None)
    p.add_argument("-a", "--algo", default="ppo")
    p.add_argument("-c", "--config", default="datt_wind_adaptive.py")
    p.add_argument("-s", "--steps", type=int, default=1000)
    p.add_argument("--viz", action="store_true", default=True)
    p.add_argument("--no-viz", action="store_false", dest="viz")
    p.add_argument("-r", "--rate", type=float, default=100)
    p.add_argument("--ref", default=None)
    p.add_argument("--seed", type=int, default=None)
    do_eval_rma(p.parse_args())
