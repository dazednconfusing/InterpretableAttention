"""
Trains new model and optionally runs specified experiments once complete

Author: Sebastian Peralta
"""
import argparse
import sys

from Transparency.ExperimentsBC import train_dataset_on_encoders
from Transparency.Trainers.DatasetBC import datasets

experiments_help = """
The following are valid experiments: ["rand_attn", "perm", "quant"]. 

"rand_attn": Runs the random attention experiment. Attention weights are 
    resampled from a gaussian with the same mean and std dev as the unrandomized distrubtion.

"perm": Runs the permutation experiment. Attention weights are randomly permuted.

"quant" Runs the quantitative analysis experiment. Positive and negative attetention scores are
    computed for each token in the training set.
"""

valid_experiments = ["rand_attn", "perm", "quant"]

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
parser.add_argument("--experiments", nargs="+", default=[], help=experiments_help)

if __name__ == "__main__":
    args, extras = parser.parse_known_args()
    args.extras = extras
    args.attention = "tanh"

    # Validate experiment keys
    if len(args.experiments) > 0 and not all(
        [expt in valid_experiments for expt in args.experiments]
    ):
        print("\nYou provided an invalid experiment key.\n")
        print(experiments_help)
        sys.exit()

    dataset = datasets[args.dataset](args)

    if args.output_dir is not None:
        dataset.output_dir = args.output_dir

    dataset.diversity = args.diversity
    encoders = [args.encoder]

    train_dataset_on_encoders(dataset, encoders, args.experiments)
