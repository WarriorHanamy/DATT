"""DATT unified CLI entry point."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="datt")
    sub = parser.add_subparsers(dest="command", required=True)

    sim_p = sub.add_parser("sim")
    sim_p.add_argument("--controller", choices=["datt", "pid", "mppi"], default="datt")
    sim_p.add_argument("-c", "--config", default="datt_hover.py")
    sim_p.add_argument("--ref", default=None)
    sim_p.add_argument("--seed", type=int, default=0)

    train_p = sub.add_parser("train")
    train_p.add_argument("-n", "--name", default=None)
    train_p.add_argument("-t", "--task", default="trajectory_fbff")
    train_p.add_argument("-a", "--algo", default="ppo")
    train_p.add_argument("-c", "--config", default="DATT_config.py")
    train_p.add_argument("--ref", default=None)
    train_p.add_argument("--seed", type=int, default=None)
    train_p.add_argument("-ts", "--timesteps", type=int, default=25_000_000)
    train_p.add_argument("--n-envs", type=int, default=10)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("-n", "--name", default=None)
    eval_p.add_argument("-t", "--task", default="trajectory_fbff")
    eval_p.add_argument("-a", "--algo", default="ppo")
    eval_p.add_argument("-c", "--config", default="datt.py")
    eval_p.add_argument("--ref", default=None)
    eval_p.add_argument("--seed", type=int, default=None)
    eval_p.add_argument("-s", "--steps", type=int, default=500)
    eval_p.add_argument("--viz", action="store_true", default=True)
    eval_p.add_argument("--no-viz", action="store_false", dest="viz")

    args = parser.parse_args()

    if args.command == "sim":
        from DATT.main import run

        run(args)
    elif args.command == "train":
        from DATT.learning.train_policy import train

        train(args)
    elif args.command == "eval":
        from DATT.learning.eval_policy import do_eval

        do_eval(args)
