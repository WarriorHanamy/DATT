import torch
import os
import numpy as np
import torch.nn.functional as F

from os.path import exists

from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.refs import TrajectoryRef

from config.configuration import AllConfig
from DATT.learning.utils.adaptation_network import AdaptationNetwork
from DATT.learning.base_env import BaseQuadsimEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from torch.utils.tensorboard import SummaryWriter


def add_e_dim(fbff_obs: np.ndarray, e: np.ndarray, base_dims=10):
    obs = np.concatenate((fbff_obs[:, :base_dims], e, fbff_obs[:, base_dims:]), axis=1)
    return obs


def remove_e_dim(output_obs: np.ndarray, e_dims: int, base_dims=10, include_extra=False):
    if include_extra:
        obs = np.concatenate(
            [output_obs[:, :base_dims], output_obs[:, (base_dims + e_dims) :]], axis=1
        )
    else:
        obs = output_obs[:, :base_dims]
    return obs


def rollout_adaptive_policy(
    rollout_len,
    adaptation_network,
    policy,
    evalenv,
    n_envs,
    time_horizon,
    base_dims,
    e_dims,
    device,
    progress=None,
):
    action_dims = 4

    history = torch.zeros((n_envs, base_dims + action_dims, time_horizon)).to(device)
    all_e_pred = None
    all_e_gt = None
    fbff_obs = remove_e_dim(evalenv.reset(), e_dims, include_extra=True)
    for i in range(rollout_len):
        e_pred = adaptation_network(history)

        input_obs = add_e_dim(fbff_obs, e_pred.detach().cpu().numpy(), base_dims)

        actions, _states = policy.predict(input_obs, deterministic=True)

        obs, rewards, dones, info = evalenv.step(actions)

        e_gt = obs[:, base_dims : (base_dims + e_dims)]
        e_gt = torch.from_numpy(e_gt).to(device).float()

        if all_e_pred is None:
            all_e_pred = e_pred
            all_e_gt = e_gt
        else:
            all_e_pred = torch.cat((all_e_pred, e_pred), dim=0)
            all_e_gt = torch.cat((all_e_gt, e_gt), dim=0)

        base_states = remove_e_dim(obs, e_dims)
        adaptation_input = np.concatenate((base_states, actions), axis=1)

        adaptation_input = torch.from_numpy(adaptation_input).to(device).float()

        history = torch.cat(
            (torch.unsqueeze(adaptation_input, -1), history[:, :, :-1].clone()), dim=2
        )

        fbff_obs = remove_e_dim(obs, e_dims, include_extra=True)

        if progress is not None:
            progress[0].update(task_id=progress[1], completed=i + 1)

    return all_e_pred, all_e_gt


def train_rma(args):
    task: DroneTask = DroneTask(args.task)
    policy_name = args.name
    if policy_name is None:
        raise ValueError("--name is required (policy name to load)")
    algo: RLAlgo = RLAlgo(args.algo)
    config_filename = args.config
    train_iterations = args.iterations
    adapt_name = args.adapt_name
    n_envs = args.n_envs

    ref_str = args.ref
    if ref_str is None:
        config_pre: AllConfig = import_config(config_filename)
        ref_str = config_pre.ref_config.default_ref
    ref = TrajectoryRef.get_by_value(ref_str)

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

    trainenv = task.env()(config=config)
    vec_env_class = SubprocVecEnv if args.subprocess else DummyVecEnv
    evalenv = make_vec_env(
        task.env(),
        n_envs=n_envs,
        vec_env_cls=vec_env_class,
        env_kwargs={
            "config": config,
            "log_scale": args.log_scale,
            "ref": ref,
            "y_max": args.ymax,
            "z_max": args.zmax,
            "diff_axis": args.diff_axis,
            "relative": args.relative,
            "body_frame": args.body_frame,
            "second_order_delay": args.second_order,
        },
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    policy = algo_class.load(SAVED_POLICY_DIR / f"{policy_name}.zip")

    action_dims = 4
    if adapt_name is not None and os.path.exists(
        SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}"
    ):
        adaptation_network_state_dict = torch.load(
            SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}",
            map_location=torch.device("cpu"),
        )
    else:
        adaptation_network_state_dict = None

    if adapt_name is None:
        adapt_name = f"{policy_name}_adapt"

    if not os.path.isdir(SAVED_POLICY_DIR / f"{policy_name}_adapt"):
        os.mkdir(SAVED_POLICY_DIR / f"{policy_name}_adapt")

    adaptation_network = AdaptationNetwork(
        input_dims=trainenv.base_dims + action_dims, e_dims=e_dims
    )
    adaptation_network = adaptation_network.to(device)
    if adaptation_network_state_dict is not None:
        adaptation_network.load_state_dict(adaptation_network_state_dict)

    optimizer = torch.optim.Adam(adaptation_network.parameters(), lr=0.001)

    writer = SummaryWriter(DEFAULT_LOG_DIR / f"{adapt_name}_logs")
    running_loss = 0.0
    try:
        from rich.progress import Progress, RateColumn, MofNCompleteColumn, TimeElapsedColumn

        pg = Progress(
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            RateColumn(unit="Iterations"),
        )
        use_rich = True
    except ImportError:
        import tqdm

        pg = None
        use_rich = False

    if use_rich:
        try:
            with pg:
                iter_task = pg.add_task("Iteration", total=train_iterations)
                rollouts_task = pg.add_task("Rollouts", total=500)

                for i in range(train_iterations):
                    optimizer.zero_grad()

                    all_e_pred, all_e_gt = rollout_adaptive_policy(
                        500,
                        adaptation_network,
                        policy,
                        evalenv,
                        n_envs,
                        time_horizon,
                        trainenv.base_dims,
                        e_dims,
                        device,
                        progress=(pg, rollouts_task),
                    )

                    loss = F.mse_loss(all_e_pred, all_e_gt)
                    loss.backward()
                    print(f"loss: {loss.detach().cpu().item()}")
                    optimizer.step()

                    running_loss += loss.detach().cpu().item()

                    if i % 500 == 0:
                        torch.save(
                            adaptation_network.state_dict(),
                            SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}_{i}",
                        )
                    if i % 10 == 0:
                        writer.add_scalar("training loss", running_loss / 10, i * 500 * n_envs)
                        running_loss = 0.0

                    pg.update(task_id=iter_task, completed=i + 1)
        finally:
            torch.save(
                adaptation_network.state_dict(),
                SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}",
            )
    else:
        for i in (bar := tqdm.tqdm(range(train_iterations), desc="rma")):
            optimizer.zero_grad()

            all_e_pred, all_e_gt = rollout_adaptive_policy(
                500,
                adaptation_network,
                policy,
                evalenv,
                n_envs,
                time_horizon,
                trainenv.base_dims,
                e_dims,
                device,
            )

            loss = F.mse_loss(all_e_pred, all_e_gt)
            loss.backward()
            print(f"loss: {loss.detach().cpu().item()}")
            optimizer.step()

            running_loss += loss.detach().cpu().item()

            if i % 500 == 0:
                torch.save(
                    adaptation_network.state_dict(),
                    SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}_{i}",
                )
            if i % 10 == 0:
                writer.add_scalar("training loss", running_loss / 10, i * 500 * n_envs)
                running_loss = 0.0

            bar.set_postfix(loss=f"{loss.detach().cpu().item():.4f}")

        torch.save(
            adaptation_network.state_dict(),
            SAVED_POLICY_DIR / f"{policy_name}_adapt" / f"{adapt_name}",
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-t", "--task", default="trajectory_fbff")
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-an", "--adapt-name", default=None)
    p.add_argument("-a", "--algo", default="ppo")
    p.add_argument("-c", "--config", default="datt_wind_adaptive.py")
    p.add_argument("-i", "--iterations", type=int, default=5000)
    p.add_argument("--ref", default=None)
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--subprocess", type=bool, default=False)
    p.add_argument("--ymax", type=float, default=0.0)
    p.add_argument("--zmax", type=float, default=0.0)
    p.add_argument("--diff-axis", type=bool, default=False)
    p.add_argument("--relative", type=bool, default=False)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--body-frame", type=bool, default=False)
    p.add_argument("--log-scale", type=bool, default=False)
    p.add_argument("--second-order", type=bool, default=False)
    p.add_argument("-de", "--device", type=int, default=0)
    train_rma(p.parse_args())
