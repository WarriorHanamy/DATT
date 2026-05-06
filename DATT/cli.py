"""DATT unified CLI entry point."""

import argparse
import sys
from os.path import getmtime

from DATT.learning.configs import SAVED_POLICY_DIR
from DATT.presets import (
    EVAL_PRESETS,
    RMA_EVAL_PRESETS,
    RMA_TRAIN_PRESETS,
    TRAIN_PRESETS,
    TRAJECTORY_CHOICES,
    TRAJECTORY_REF_MAP,
)


def _find_latest_checkpoint():
    zips = sorted(SAVED_POLICY_DIR.glob("*.zip"), key=getmtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No checkpoints found in {SAVED_POLICY_DIR}")
    return zips[0].stem


def main():
    parser = argparse.ArgumentParser(
        prog="datt",
        description="DATT: Deep Adaptive Trajectory Tracking for quadrotor control.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = sub.add_parser("train", help="Train a DATT policy")
    train_p.add_argument(
        "-p",
        "--preset",
        choices=list(TRAIN_PRESETS.keys()),
        default="tracking",
        help="Training scenario: hover | tracking | wind (default: tracking)",
    )
    train_p.add_argument(
        "-n", "--name", default=None, help="Custom save name (default: auto-generated)"
    )
    train_p.add_argument(
        "-t",
        "--trajectory",
        choices=TRAJECTORY_CHOICES,
        default=None,
        help="Reference trajectory for training (default: config-defined)",
    )
    train_p.add_argument("--seed", type=int, default=None, help="Random seed")
    train_p.add_argument(
        "-s", "--timesteps", type=int, default=25_000_000, help="Total training steps"
    )
    train_p.add_argument("-e", "--envs", type=int, default=10, help="Parallel environments")

    # ---- eval ----
    eval_p = sub.add_parser("eval", help="Evaluate a trained or pretrained policy")
    eval_p.add_argument(
        "-p",
        "--policy",
        choices=["latest"] + list(EVAL_PRESETS.keys()),
        nargs="?",
        const="latest",
        default="latest",
        help=(
            "Policy to evaluate. "
            "latest = auto-find most recent local checkpoint. "
            "Named values = pretrained checkpoints: hover (datt_hover), tracking (datt), wind (datt_wind_adaptive). "
            "(default: latest)"
        ),
    )
    eval_p.add_argument(
        "-t",
        "--trajectory",
        choices=TRAJECTORY_CHOICES,
        default="zigzag",
        help="Reference trajectory to track (default: zigzag)",
    )
    eval_p.add_argument("--seed", type=int, default=None, help="Random seed")
    eval_p.add_argument("-s", "--steps", type=int, default=500, help="Steps per episode")
    eval_p.add_argument(
        "--config",
        default=None,
        help="Override config file (advanced; usually auto-selected by --policy)",
    )
    eval_p.add_argument("--viz", action="store_true", default=False, help="Show matplotlib plots")

    # ---- train-rma ----
    train_rma_p = sub.add_parser("train-rma", help="Train an RMA adaptation network")
    train_rma_p.add_argument("-n", "--name", default=None, help="Base policy name (required)")
    train_rma_p.add_argument(
        "-p",
        "--preset",
        choices=list(RMA_TRAIN_PRESETS.keys()),
        default="wind",
        help="Training scenario (default: wind)",
    )
    train_rma_p.add_argument(
        "-an", "--adapt-name", default=None, help="Adaptation network save name"
    )
    train_rma_p.add_argument(
        "-i", "--iterations", type=int, default=5000, help="RMA training iterations"
    )
    train_rma_p.add_argument("-e", "--envs", type=int, default=10, help="Parallel environments")
    train_rma_p.add_argument(
        "-t",
        "--trajectory",
        choices=TRAJECTORY_CHOICES,
        default=None,
        help="Reference trajectory (default: config-defined)",
    )
    train_rma_p.add_argument("--seed", type=int, default=None, help="Random seed")
    train_rma_p.add_argument("--subprocess", type=bool, default=False)
    train_rma_p.add_argument("--ymax", type=float, default=0.0)
    train_rma_p.add_argument("--zmax", type=float, default=0.0)
    train_rma_p.add_argument("--diff-axis", type=bool, default=False)
    train_rma_p.add_argument("--relative", type=bool, default=False)
    train_rma_p.add_argument("--body-frame", type=bool, default=False)
    train_rma_p.add_argument("--log-scale", type=bool, default=False)
    train_rma_p.add_argument("--second-order", type=bool, default=False)
    train_rma_p.add_argument("-de", "--device", type=int, default=0, help="CUDA device index")

    # ---- eval-rma ----
    eval_rma_p = sub.add_parser("eval-rma", help="Evaluate a policy with RMA adaptation")
    eval_rma_p.add_argument(
        "-p",
        "--policy",
        choices=list(RMA_EVAL_PRESETS.keys()),
        default="wind",
        help="Policy + adaptation to evaluate (default: wind)",
    )
    eval_rma_p.add_argument(
        "-an",
        "--adapt-name",
        default=None,
        help="Override adaptation network name (default: auto from --policy)",
    )
    eval_rma_p.add_argument(
        "-t",
        "--trajectory",
        choices=TRAJECTORY_CHOICES,
        default="zigzag",
        help="Reference trajectory to track (default: zigzag)",
    )
    eval_rma_p.add_argument("--seed", type=int, default=None, help="Random seed")
    eval_rma_p.add_argument("-s", "--steps", type=int, default=1000, help="Steps per episode")
    eval_rma_p.add_argument("-r", "--rate", type=float, default=100, help="Render rate (Hz)")
    eval_rma_p.add_argument(
        "--viz", action="store_true", default=False, help="Show matplotlib plots"
    )

    args = parser.parse_args()

    # ---- dispatch ----

    if args.command == "train":
        preset = TRAIN_PRESETS[args.preset]
        args.task = preset["task"]
        args.algo = preset["algo"]
        args.config = preset["config"]
        args.ref = TRAJECTORY_REF_MAP.get(args.trajectory) if args.trajectory else None

        from DATT.learning.train_policy import train

        train(args)

    elif args.command == "eval":
        if args.policy == "latest":
            args.name = _find_latest_checkpoint()
            args.task = "trajectory_fbff"
            args.algo = "ppo"
            if args.config is None:
                args.config = "datt.py"
            print(f"Using latest checkpoint: {args.name}")
        else:
            preset = EVAL_PRESETS[args.policy]
            args.name = preset["checkpoint"]
            args.task = preset["task"]
            args.algo = preset["algo"]
            if args.config is None:
                args.config = preset["config"]

        args.ref = TRAJECTORY_REF_MAP[args.trajectory]

        from DATT.learning.eval_policy import do_eval

        do_eval(args)

    elif args.command == "train-rma":
        if args.name is None:
            parser.error("train-rma: --name/-n is required (base policy checkpoint name)")

        preset = RMA_TRAIN_PRESETS[args.preset]
        args.task = preset["task"]
        args.algo = preset["algo"]
        args.config = preset["config"]
        args.ref = TRAJECTORY_REF_MAP.get(args.trajectory) if args.trajectory else None

        from DATT.learning.train_rma import train_rma

        train_rma(args)

    elif args.command == "eval-rma":
        preset = RMA_EVAL_PRESETS[args.policy]
        args.name = preset["checkpoint"]
        args.task = preset["task"]
        args.algo = preset["algo"]
        args.config = preset["config"]
        if args.adapt_name is None:
            args.adapt_name = preset["adapt_name"]
        args.ref = TRAJECTORY_REF_MAP[args.trajectory]

        from DATT.learning.eval_rma_policy import do_eval_rma

        do_eval_rma(args)
