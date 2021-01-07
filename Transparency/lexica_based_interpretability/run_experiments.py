"""Runs Experiments Specified in Arguments

Author: Sebastian Peralta
"""
import argparse
import os
import sys

from Transparency.ExperimentsBC import run_experiments_on_latest_model, train_dataset
from Transparency.train_and_run_experiments_bc import (
    experiments_help,
    valid_experiments,
)
from Transparency.Trainers.DatasetBC import datasets

parser = argparse.ArgumentParser(description="Run experiments on a dataset")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str)
parser.add_argument(
    "--encoder",
    type=str,
    choices=["vanilla_lstm", "ortho_lstm", "diversity_lstm"],
    required=True,
)
parser.add_argument("--diversity", type=float, default=0)
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
    help="Set this option to train a new model and run all supported specified experiments every epoch.",
)
parser.add_argument("--experiments", nargs="+", default=[], help=experiments_help)

if __name__ == "__main__":
    args, extras = parser.parse_known_args()
    args.extras = extras
    args.attention = "tanh"

    dataset = datasets[args.dataset](args)

    if args.output_dir is not None:
        dataset.output_dir = args.output_dir

    # Validate experiment keys
    if len(args.experiments) > 0 and not all(
        [expt in valid_experiments for expt in args.experiments]
    ):
        print("\nYou provided an invalid experiment key.\n")
        print(experiments_help)
        sys.exit()

    dataset.diversity = args.diversity
    encoders = [args.encoder]

    for e in encoders:
        if args.train:
            print(" SOMEHOW ITS TRAIN TRUE")
            train_dataset(dataset, e, args.experiments)

        run_experiments_on_latest_model(dataset, e, args.experiments)
