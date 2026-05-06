import sys
import time
import numpy as np

import DATT

sys.modules["quadsim"] = DATT

import matplotlib.pyplot as plt

from os.path import exists, getmtime
from argparse import ArgumentParser

from DATT.learning.configs import *
from DATT.learning.tasks import DroneTask
from DATT.refs import TrajectoryRef

from DATT.quadsim.visualizer import Vis

from scipy.spatial.transform import Rotation as R

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from config.configuration import AllConfig

from DATT.learning.adaptation_module import Adapation


def _find_latest_checkpoint():
    """Return the basename (without .zip) of the most recently modified checkpoint."""
    zips = sorted(SAVED_POLICY_DIR.glob("*.zip"), key=getmtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No checkpoints found in {SAVED_POLICY_DIR}")
    return zips[0].stem


def do_eval(args):
    task: DroneTask = DroneTask(args.task)
    algo: RLAlgo = RLAlgo(args.algo)
    config_filename = args.config

    # resolve policy name
    policy_name = args.name
    if policy_name is None:
        policy_name = _find_latest_checkpoint()
        print(f"Auto-loading latest checkpoint: {policy_name}")

    if not exists(SAVED_POLICY_DIR / f"{policy_name}.zip"):
        raise ValueError(f"policy not found: {policy_name}")
    if not exists(CONFIG_DIR / config_filename):
        raise FileNotFoundError(f"{config_filename} is not a valid config file")

    config: AllConfig = import_config(config_filename)

    # resolve trajectory reference
    ref_str = args.ref
    if ref_str is None:
        ref_str = config.ref_config.default_ref
    ref = TrajectoryRef.get_by_value(ref_str)

    eval_steps = args.steps
    viz = args.viz

    l1_sim = config.sim_config.L1_simulation

    if task.is_trajectory():
        seed = args.seed
        if seed is None:
            seed = np.random.randint(0, 100000)
            fixed_seed = False
        else:
            fixed_seed = True
        print(seed)
        evalenv = task.env()(config=config, ref=ref, seed=seed, fixed_seed=fixed_seed)
    else:
        evalenv = task.env()(config=config)

    algo_class = algo.algo_class()
    policy = algo_class.load(SAVED_POLICY_DIR / f"{policy_name}.zip")

    control_error_avg = 0
    adaptation_module = Adapation()
    count = 0
    if viz:
        vis = Vis()
    try:
        while True:
            count += 1
            total_r = 0
            obs = evalenv.reset()
            adaptation_module.reset()
            all_states = []
            all_rewards = []
            all_actions = []
            all_ang_vel_actual = []
            all_ang_vel_desired = []
            des_traj = []
            control_errors = []
            all_wind = []
            l1_terms = []
            try:
                print("wind field", evalenv.wind_vector)
                print("mass", evalenv.model.mass)
                print("k", evalenv.k)
            except:
                pass
            for i in range(eval_steps):
                action, _states = policy.predict(obs, deterministic=True)
                act = action
                obs, rewards, dones, info = evalenv.step(act)
                state = evalenv.getstate()

                wind_vector = evalenv.wind_vector
                all_wind.append(wind_vector.copy())
                if l1_sim:
                    l1_terms.append(evalenv.adaptation_module.d_hat.copy())
                all_actions.append(action)
                all_rewards.append(rewards)
                all_states.append(np.r_[state.pos, state.vel, obs[6:10]])
                try:
                    des_traj.append(evalenv.ref.pos(evalenv.t))
                except:
                    pass
                total_r += rewards
                if task.is_trajectory():
                    control_error = np.linalg.norm(state.pos - evalenv.ref.pos(evalenv.t))
                    control_errors.append(control_error)
                if viz:
                    vis.set_state(state.pos.copy(), state.rot)
                time.sleep(0.01)

            all_wind = np.array(all_wind)
            l1_terms = np.array(l1_terms)
            all_states = np.array(all_states)
            all_actions = np.array(all_actions)
            all_rewards = np.array(all_rewards)
            all_ang_vel_desired = np.array(all_ang_vel_desired)
            all_ang_vel_actual = np.array(all_ang_vel_actual)
            des_traj = np.array(des_traj)

            if viz:
                plt.figure()
                ax = plt.subplot(3, 1, 1)
                plt.plot(range(eval_steps), all_states[:, 0])
                if des_traj.size > 0:
                    plt.plot(range(eval_steps), des_traj[:, 0])
                plt.subplot(3, 1, 2)
                plt.plot(range(eval_steps), all_states[:, 1])
                if des_traj.size > 0:
                    plt.plot(range(eval_steps), des_traj[:, 1])
                plt.subplot(3, 1, 3)
                plt.plot(range(eval_steps), all_states[:, 2])
                if des_traj.size > 0:
                    plt.plot(range(eval_steps), des_traj[:, 2])
                plt.suptitle("PPO (sim) des vs. actual pos mass : {}".format(evalenv.model.mass))

                plt.figure()
                ax = plt.subplot(3, 1, 1)
                plt.plot(range(eval_steps), all_wind[:, 0], label="x")
                if l1_sim:
                    plt.plot(range(eval_steps), l1_terms[:, 0], label="L1 x")
                plt.subplot(3, 1, 2, sharex=ax)
                plt.plot(range(eval_steps), all_wind[:, 1], label="y")
                if l1_sim:
                    plt.plot(range(eval_steps), l1_terms[:, 1], label="L1 y")
                plt.subplot(3, 1, 3, sharex=ax)
                plt.plot(range(eval_steps), all_wind[:, 2], label="z")
                if l1_sim:
                    plt.plot(range(eval_steps), l1_terms[:, 2], label="L1 z")
                plt.suptitle("L1 vs. Wind")

                try:
                    plt.figure()
                    plt.plot(all_states[:, 0], all_states[:, 1], label="actual")
                    plt.plot(des_traj[:, 0], des_traj[:, 1], label="desired")
                    plt.legend()
                except:
                    pass

                eulers = np.array(
                    [R.from_quat(rot).as_euler("ZYX")[::-1] for rot in all_states[:, 6:10]]
                )

                plt.show()
            print(total_r)
            if control_errors:
                control_error_avg += np.mean(control_errors)
                print("control error", np.mean(control_errors))
    except KeyboardInterrupt:
        if control_error_avg > 0:
            print("Control Error: ", control_error_avg / count)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-t", "--task", default="trajectory_fbff")
    p.add_argument("-a", "--algo", default="ppo")
    p.add_argument("-c", "--config", default="datt.py")
    p.add_argument("--ref", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-s", "--steps", type=int, default=500)
    p.add_argument("--viz", type=bool, default=True)
    do_eval(p.parse_args())
