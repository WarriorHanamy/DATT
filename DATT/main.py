import sys

import numpy as np

from DATT.quadsim.sim import QuadSim
from DATT.quadsim.models import IdentityModel

from DATT.learning.configs import *
from DATT.controllers import cntrl_config_presets, ControllersZoo
from config.configuration import AllConfig
from DATT.refs import TrajectoryRef
from DATT.python_utils.plotu import subplot, set_3daxes_equal

import matplotlib.pyplot as plt


def run(args):
    config: AllConfig = import_config(args.config)

    # resolve trajectory reference
    if args.ref is None:
        default_ref = config.ref_config.default_ref
    else:
        default_ref = args.ref
    ref_type = TrajectoryRef.get_by_value(default_ref)

    dt = config.sim_config.dt()
    vis = True
    plot = True
    t_end = 10.0

    ref = ref_type.ref(
        config.ref_config,
        seed=args.seed,
        env_diff_seed=config.training_config.env_diff_seed,
    )

    model = IdentityModel()
    quadsim = QuadSim(model, vis=vis)

    # derive controller config preset from controller name
    preset_name = f"{args.controller}_config"
    cntrl_config = getattr(cntrl_config_presets, preset_name, None)
    if cntrl_config is None:
        cntrl_config = getattr(cntrl_config_presets, "datt_hover_config")
    controller_type = ControllersZoo(args.controller)
    controller = controller_type.cntrl(config, {controller_type._value_: cntrl_config})
    controller.ref_func = ref

    dists = []
    ts = quadsim.simulate(dt=dt, t_end=t_end, controller=controller, dists=dists)

    if not plot:
        sys.exit(0)

    eulers = np.array([rot.as_euler("ZYX")[::-1] for rot in ts.rot])

    plt.figure()
    ax = plt.subplot(3, 1, 1)
    plt.plot(ts.times, ts.pos[:, 0])
    plt.plot(ts.times, ref.pos(ts.times)[0, :])
    plt.subplot(3, 1, 2, sharex=ax)
    plt.plot(ts.times, ts.pos[:, 1])
    plt.plot(ts.times, ref.pos(ts.times)[1, :])
    plt.subplot(3, 1, 3, sharex=ax)
    plt.plot(ts.times, ts.pos[:, 2])
    plt.plot(ts.times, ref.pos(ts.times)[2, :])
    plt.suptitle(type(controller).__name__)

    plt.figure()
    plt.plot(ts.pos[:, 0], ts.pos[:, 1], label="actual")
    plt.legend()

    subplot(ts.times, ts.vel, yname="Vel. (m)", title="Velocity")
    subplot(
        ts.times,
        ref.vel(ts.times).T,
        yname="Vel. (m)",
        title="Velocity",
        label="Desired",
    )
    subplot(ts.times, eulers, yname="Euler (rad)", title="ZYX Euler Angles")
    subplot(ts.times, ts.ang, yname="$\\omega$ (rad/s)", title="Angular Velocity")
    subplot(ts.times, ts.force, yname="Force (N)", title="Body Z Thrust")
    plt.show()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["datt", "pid", "mppi"], default="datt")
    p.add_argument("-c", "--config", default="datt_hover.py")
    p.add_argument("--ref", default=None)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
