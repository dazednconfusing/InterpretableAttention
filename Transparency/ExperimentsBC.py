from Transparency.common_code.common import *
from Transparency.configurations import configurations
from Transparency.Trainers.PlottingBC import generate_graphs
from Transparency.Trainers.TrainerBC import Evaluator, Trainer


def train_dataset(dataset, config="lstm", experiments=None):
    """Configures dataset and trains model with new Trainer object

    Args:
        dataset (Dataset): Dataset wrapper which includes metadata
        config (str, optional): Can be "vanilla", "lstm", or "diversity". Defaults to "lstm".
        rand_attn_expt (bool, optional): Runs the random attention experiment every epoch. Defaults to False.
        perm_expt (bool, optional): Runs the permutation experiment every epoch.. Defaults to False.

    Returns:
        Trainer, Evaluater: Tuple of two wrapper objects that can be used to train and evaluate a model.
    """

    config = configurations[config](dataset)
    trainer = Trainer(
        dataset, config=config, _type=dataset.trainer_type, experiments=experiments
    )
    if hasattr(dataset, "n_iter"):
        n_iters = dataset.n_iter
    else:
        n_iters = 8

    trainer.train(
        dataset.train_data,
        dataset.dev_data,
        n_iters=n_iters,
        save_on_metric=dataset.save_on_metric,
    )
    evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return trainer, evaluator


def train_dataset_on_encoders(dataset, encoders, experiments=None):
    """Trains model on dataset and runs experiments immediately after.

    Args:
        dataset (Dataset): Dataset wrapper.
        encoders (list<string>): List of encoder keys. Can be "lstm", "diversity", "vanilla"
        experiments (list<string>, optional): List of experiment keys. Defaults to None.
    """
    for e in encoders:
        train_dataset(dataset, e)
        run_experiments_on_latest_model(dataset, e, experiments)


def run_evaluator_on_latest_model(dataset, config="lstm"):
    config = configurations[config](dataset)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator


def run_experiments_on_latest_model(
    dataset, config="lstm", force_run=True, experiments=None
):
    """Runs the experiments specified on the latest model.
    If none are specified, only quantiitative analysis is run.

    Args:
        dataset (Dataset): Dataset Wrapper
        config (str, optional): Can be "vanilla", "lstm", or "diversity". Defaults to "lstm".
        force_run (bool, optional): Defaults to True.
        experiments (list<string>, optional): Specify what experiments you want to run.
            Can be one of the valid experiments in configuations.py. Defaults to None.

    Author: Sebastian Peralta
    """
    evaluator = run_evaluator_on_latest_model(dataset, config)
    test_data = dataset.test_data
    experiments_ = {
        "rand_attn": evaluator.random_attn_experiment,
        "perm": evaluator.permutation_experiment,
        "quant": evaluator.quantitative_analysis_experiment,
        # "importance": evaluator.importance_ranking_experiment,
        # "conicity": evaluator.conicity_analysis_experiment,
        # "grad": evaluator.gradient_experiment,
        # "integrad": evaluator.integrated_gradient_experiment,
    }

    if experiments is None:
        evaluator.quantitative_analysis_experiment(
            test_data, dataset, force_run=force_run
        )
    else:
        for expt in experiments:
            if expt == "conicity":
                experiments_[expt](test_data)

            experiments_[expt](test_data, force_run=force_run)

    # Comment out experiments you don't want to run here
    # evaluator.gradient_experiment(test_data, force_run=force_run)
    # evaluator.importance_ranking_experiment(test_data, force_run=force_run)
    # evaluator.conicity_analysis_experiment(test_data)
    # evaluator.permutation_experiment(test_data, force_run=force_run)
    # evaluator.integrated_gradient_experiment(dataset, force_run=force_run)


def generate_graphs_on_latest_model(dataset, config="lstm"):

    config = configurations[config](dataset)
    latest_model = get_latest_model(
        os.path.join(config["training"]["basepath"], config["training"]["exp_dirname"])
    )
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=False)
    generate_graphs(
        dataset,
        config["training"]["exp_dirname"],
        evaluator.model,
        test_data=dataset.test_data,
    )
