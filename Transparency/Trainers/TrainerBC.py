import copy
import csv
import datetime

import numpy as np
import Transparency.model.Binary_Classification as BC
from Transparency.common_code.common import *
from Transparency.common_code.metrics import *

metrics_type = {
    "Single_Label": calc_metrics_classification,
    "Multi_Label": calc_metrics_multilabel,
}

epoch_accuracies_filename = "epoch_test_accuracies.csv"


class Trainer:
    """Wrapper Object to train and run experiments on a model"""

    def __init__(self, dataset, config, _type="Single_Label", experiments=None):
        """

        Args:
            dataset (numpy.array): [description]
            config (dict): Configuration object for model
            experiments (list): experiments to be run every epoch
            _type (str, optional): Type of data. Defaults to "Single_Label".

        Author: Sebastian Peralta
        """
        Model = BC.Model
        self.model = Model(config, pre_embed=dataset.vec.embeddings)
        self.metrics = metrics_type[_type]
        self.display_metrics = True
        self.train_experiments = {
            "rand_attn": self.random_attn_experiment,
            "perm": self.permutation_experiment,
        }
        self.experiments = [] if experiments is None else experiments

    def train(self, train_data, test_data, n_iters=8, save_on_metric="roc_auc"):
        best_metric = 0.0
        dirname = self.model.save_values(save_model=False)

        # Author: Sebastian Peralta
        # Save epoch accuracies to csv
        with open(dirname + "/" + epoch_accuracies_filename, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "accuracy"])

        for i in range(n_iters):

            print("Starting Epoch: {}".format(i))

            self.model.train(train_data.X, train_data.y, epoch=i)
            predictions, _, conicity_values = self.model.evaluate(test_data.X)
            predictions = np.array(predictions)
            test_metrics = self.metrics(test_data.y, predictions)

            # Author: Sebastian Peralta
            for expt in self.experiments:
                self.train_experiments[expt](test_data, dirname, epoch=i)

            if conicity_values is not None:
                test_metrics["conicity_mean"] = np.mean(conicity_values)
                test_metrics["conicity_std"] = np.std(conicity_values)

            if self.display_metrics:
                print_metrics(test_metrics)

            metric = test_metrics[save_on_metric]

            if i == 0:
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)

            elif metric > best_metric and i > 0:
                best_metric = metric
                save_model = True
                print("Model Saved on ", save_on_metric, metric)
            else:
                save_model = False
                print("Model not saved on ", save_on_metric, metric)

            dirname = self.model.save_values(save_model=save_model)
            print(dirname)
            f = open(dirname + "/epoch.txt", "a")
            f.write(str(test_metrics) + "\n")
            f.close()

            # Author: Sebastian Peralta
            with open(dirname + "/" + epoch_accuracies_filename, "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([i, test_metrics["accuracy"]])

    def permutation_experiment(self, test_data, dirname, epoch):
        """Runs permutation experiment on model. The attention weights are randomly permuted and test accuracy saved

        Args:
            test_data (np.array): Test data
            dirname (string): Directory name to store csv
            epoch (int): Epoch

        Author: Sebastian Peralta
        """
        pref = "perm_expt_"
        if epoch == 0:
            with open(
                dirname + "/" + pref + epoch_accuracies_filename, "w+", newline=""
            ) as h:
                writer = csv.writer(h)
                writer.writerow(["epoch", "accuracy"])
        with torch.no_grad():
            preds = self.model.permute_attn(test_data.X, num_perm=10)
            metrics = self.metrics(test_data.y, preds)

            with open(
                dirname + "/" + pref + epoch_accuracies_filename, "w+", newline=""
            ) as h:
                writer = csv.writer(h)
                writer.writerow([epoch, metrics["accuracy"]])

    def random_attn_experiment(self, test_data, dirname, epoch):
        """Runs random attention experiment. All attention weights are redrawn from a gaussian with the same mean
        and std deviation and test accuracy saved.

        Args:
            test_data (np.array): Test data
            dirname (string): Directory name to store csv
            epoch (int): Epoch

        Author: Sebastian Peralta
        """
        pref = "rand_attn_expt_"
        if epoch == 0:
            with open(
                dirname + "/" + pref + epoch_accuracies_filename, "w+", newline=""
            ) as h:
                writer = csv.writer(h)
                writer.writerow(["epoch", "accuracy"])
        with torch.no_grad():
            model = copy.deepcopy(self.model)
            model.randomize_attn()

            preds, _, _ = model.evaluate(test_data.X)
            preds = np.array(preds)
            metrics = self.metrics(test_data.y, preds)

        with open(
            dirname + "/" + pref + epoch_accuracies_filename, "w+", newline=""
        ) as h:
            writer = csv.writer(h)
            writer.writerow([epoch, metrics["accuracy"]])


class Evaluator:
    def __init__(self, dataset, dirname, _type="Single_Label"):
        Model = BC.Model
        self.model = Model.init_from_config(dirname, load_gen=False)
        self.model.dirname = dirname
        self.metrics = metrics_type[_type]
        self.display_metrics = True
        self.dataset = dataset

    def evaluate(self, test_data, save_results=False):
        predictions, attentions, conicity_values = self.model.evaluate(test_data.X)
        predictions = np.array(predictions)

        test_metrics = self.metrics(test_data.y, predictions)

        if conicity_values is not None:
            test_metrics["conicity_mean"] = str(np.mean(conicity_values))
            test_metrics["conicity_std"] = str(np.std(conicity_values))

        if self.display_metrics:
            print_metrics(test_metrics)

        if save_results:
            f = open(self.model.dirname + "/evaluate.json", "w")
            json.dump(test_metrics, f)
            f.close()

        test_data.yt_hat = predictions
        test_data.attn_hat = attentions

        test_output = {
            "X": test_data.X,
            "y": test_data.y,
            "yt_hat": test_data.yt_hat,
            "attn_hat": test_data.attn_hat,
        }
        pdump(self.model, test_output, "test_output")

        return predictions, attentions

    # IG helpers
    def get_grads_from_custom_td(self, test_data):
        print("getting normal grads")
        grads = self.model.gradient_mem(test_data)
        return grads

    def evaluate_outputs_from_embeds(self, embds):
        predictions, attentions = self.model.evaluate(embds)
        return predictions, attentions

    def evaluate_outputs_from_custom_td(self, testdata):
        predictions, _ = self.model.evaluate(testdata)
        return predictions

    def importance_ranking_experiment(self, test_data, force_run=False):
        if force_run or not is_pdumped(self.model, "importance_ranking"):
            print("Running Importance Ranking Expt ...")
            importance_ranking = self.model.importance_ranking(test_data.X)
            print("Dumping Importance Ranking Outputs")
            pdump(self.model, importance_ranking, "importance_ranking")

    def permutation_experiment(self, test_data, force_run=False):
        """Runs permutation experiment on model. The attention weights are randomly permuted and test results saved.

        Args:
            test_data (np.Array): Test data
            force_run (bool, optional): Defaults to False.

        Author: Sebastian Peralta
        """
        if force_run or not is_pdumped(self.model, "permutations"):
            print("Running Permutation Expt ...")
            perms = self.model.permute_attn(test_data.X, num_perm=10)
            print("Dumping Permutation Outputs")
            pdump(self.model, perms, "permutations")

    def random_attn_experiment(self, test_data, force_run=False):
        """Runs random attention experiment. All attention weights are redrawn from a
        gaussian with the same mean and std dev. Test Results are saved.


        Args:
            test_data (np.array): Test data
            force_run (bool, optional): Defaults to False.

        Author: Sebastian Peralta
        """
        if force_run or not is_pdumped(self.model, "random_attns"):
            print("Running Random Attn Expt ...")
            new_model = copy.deepcopy(self.model)
            new_model.randomize_attn()

            new_predictions, _, _ = new_model.evaluate(test_data.X)
            new_predictions = np.array(new_predictions)
            new_test_metrics = self.metrics(test_data.y, new_predictions)

            print("Dumping Random Attn Outputs")
            pdump(new_model, new_test_metrics, "random_attns")

    def conicity_analysis_experiment(self, test_data):
        self.model.conicity_analysis(test_data.X)

    def integrated_gradient_experiment(self, dataset, force_run=False):
        if force_run or not is_pdumped(self.model, "integrated_gradients"):
            print("Running Integrated Gradients Expt ...")
            int_grads = self.model.integrated_gradient_mem(dataset)
            print("Dumping Integrated Gradients Outputs")
            pdump(self.model, int_grads, "integrated_gradients")

    def quantitative_analysis_experiment(self, test_data, dataset, force_run=False):
        if force_run or not is_pdumped(self.model, "quant_analysis"):
            print("Running Analysis by Parts-of-speech Expt ...")
            quant_output = self.model.quantitative_analysis(
                test_data.X, test_data.y, dataset
            )
            print("Dumping Parts-of-speech Expt Outputs")
            pdump(self.model, quant_output, "quant_analysis")

    def gradient_experiment(self, test_data, force_run=False):
        if force_run or not is_pdumped(self.model, "gradients"):
            print("Running Gradients Expt ...")
            grads = self.model.gradient_mem(test_data.X)[0]
            print("Dumping Gradients Outputs")
            pdump(self.model, grads, "gradients")
