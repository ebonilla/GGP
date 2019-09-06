# Author: Yin Cheng Ng

from ggp.utils import *
from ggp.kernels import SparseGraphPolynomial
from ggp.model import GraphSVGP
from scipy.cluster.vq import kmeans2
import numpy as np
import os, time, pickle, argparse
import networkx as nx
import pandas as pd
import csv

import click

class SSLExperimentNoisy(object):

    def __init__(self, data_name, adj_name,  random_seed_np, random_seed_tf, random_split, split_sizes, random_split_seed,
                 add_val, add_val_seed, p, results_dir):
        self.data_name = data_name.lower()
        self.random_seed = int(random_seed_np)
        np.random.seed(self.random_seed)
        tf.set_random_seed(random_seed_tf)

        self.adj_name = adj_name
        self.results_dir = results_dir

        # Load data
        self.adj_mat, self.node_features, self.x_tr, self.y_tr, self.x_val, self.y_val, self.x_test, self.y_test \
            = load_data_ssl(data_name=self.data_name,
                            random_split=random_split,
                            split_sizes=split_sizes,
                            random_split_seed=random_split_seed,
                            add_val=add_val,
                            add_val_seed=add_val_seed,
                            p_val=p)

        # loads noisy adjacency
        self.adj_mat_noisy = self.load_noisy_adjacency(self.adj_name)

        # Init kernel
        k = SparseGraphPolynomial(self.adj_mat_noisy, self.node_features, self.x_tr, degree=3.)
        k.offset = np.abs(np.random.randn(1) + 5.); k.offset.fixed = False
        k.variance = 1.; k.variance.fixed = True
        # Init inducing points
        ind_points = kmeans2(self.node_features, len(self.x_tr), minit='points')[0]
        # Init optimizer
        self.optimizer = tf.train.AdamOptimizer(0.0005)
        # Init model
        self.m = GraphSVGP(self.x_tr, self.y_tr, k, GPflow.likelihoods.MultiClass(len(np.unique(self.y_tr))), ind_points,
                      num_latent=len(np.unique(self.y_tr)), minibatch_size=len(self.x_tr), whiten=True, q_diag=False)
        # Define housekeeping variables
        self.last_ts = time.time()
        self.iter = 0; self.check_obj_every = 200
        self.log_iter = []; self.log_t = []; self.log_obj = []; self.log_param = None; self.log_opt_state = None;
        self.param_fp = os.path.join(self.results_dir, 'ssl_param_files')
        if not (os.path.isdir(self.param_fp)):
            os.mkdir(self.param_fp)

        #self.param_fp = os.path.join(self.param_fp, 'SSL-{0}-rs_{1}.p'.format(self.data_name, random_seed))
        self.param_fp = os.path.join(self.param_fp, os.path.basename(adj_name) + '-rs_{0}.p'.format(self.random_seed))

        self.m._compile(self.optimizer)
        if os.path.isfile(self.param_fp):
            print 'Param. file already exists! Loading from {0}.'.format(self.param_fp)
            self.load_snapshot(self.param_fp)
        else:
            self.save_snapshot(self.param_fp, update_before_saving=True)

    def load_noisy_adjacency(self, adjacency_matrix_fname):
        g_ = nx.read_gpickle(adjacency_matrix_fname)
        A = (nx.adjacency_matrix(g_).toarray()).astype(dtype=np.float32)
        return A

    def update_log(self, param):
        self.log_t.append(time.time() - self.last_ts)
        self.log_param = param.copy()
        self.log_opt_state = self.m.get_optimizer_variables()[0]
        self.log_obj.append(self.m._objective(param)[0])
        self.log_iter.append(self.iter);
        self.m.set_state(param)

    def save_snapshot(self, pickle_fp, update_before_saving = False):
        log_up_to_date = self.iter >= self.log_iter[-1] if len(self.log_iter)>1 else False
        if update_before_saving and not(log_up_to_date):
            param = self.m.get_free_state()
            self.update_log(param)
        p_dict = {}
        p_dict['iter'] = self.iter; p_dict['log_iter'] = self.log_iter; p_dict['log_t'] = self.log_t;
        p_dict['log_obj'] = self.log_obj;
        p_dict['log_opt_state'] = self.log_opt_state; p_dict['log_param'] = self.log_param;
        pickle.dump(p_dict, open(pickle_fp, "wb"))

    def load_snapshot(self, pickle_fp):
        p_dict = pickle.load(open(pickle_fp, 'rb'))
        self.iter = p_dict['iter'];
        self.log_iter = p_dict['log_iter'];
        self.log_t = p_dict['log_t'];
        self.log_param = p_dict['log_param'];
        self.log_opt_state = p_dict['log_opt_state']
        self.log_obj = p_dict['log_obj'];
        self.m.set_optimizer_variables_value(self.log_opt_state.copy())
        self.m.set_state(self.log_param.copy())

    def _callback(self, param):
        self.iter += 1
        if (self.iter % self.check_obj_every) == 0:
            self.update_log(param)
            self.last_ts = time.time()
            print 'SSL-{0}-rs_{1}'.format(self.data_name, self.random_seed), self.log_iter[-1], self.log_obj[-1], \
                 '({0:.3f}s)'.format(self.log_t[-1])
            # Save a snapshot of the model with the best ELBO
            if self.log_obj[-1] < np.min(np.array(self.log_obj)[:-1]):
                self.save_snapshot(self.param_fp)

    def train(self, maxiter, check_obj_every_n_iter = None):
        self.check_obj_every = self.check_obj_every if check_obj_every_n_iter is None else check_obj_every_n_iter
        if self.iter < maxiter:
            self.last_ts = time.time()
            self.m.optimize(method=self.optimizer, maxiter=maxiter - self.iter, callback=self._callback)
        print '{0} iterations completed.'.format(self.iter)

    def evaluate(self, results_dir):
        print '\nEvaluating prediction accuracies...'
        # Restore the parameters to the one with the best ELBO
        tmp_params = self.m.get_free_state().copy()
        self.m.set_state(pickle.load(open(self.param_fp, 'rb'))['log_param'])
        pred_train = self.m.predict_y(self.x_tr)[0]
        pred_test = self.m.predict_y(self.x_test)[0]
        ytrain =  self.y_tr.flatten()
        ytest = self.y_test.flatten()

        tr_acc = evaluate_accuracy(ytrain, pred_train)
        test_acc = evaluate_accuracy(ytest, pred_test)

        tr_mnlp = evaluate_mnlp(ytrain, pred_train)
        test_mnlp = evaluate_mnlp(ytest, pred_test)

        print 'Prediction metrics: '
        print '\tTraining Accuracy: {0:.4f}'.format(tr_acc)
        print '\tTest Accuracy: {0:.4f}'.format(test_acc)
        print '\tTraining MNLP: {0:.4f}'.format(tr_mnlp)
        print '\tTest MNLP: {0:.4f}'.format(test_mnlp)

        #fout = os.path.basename(self.adj_name) + "_seed_" + str(self.random_seed)

        write_test_predictions(ytest, pred_test, test_acc, test_mnlp, results_dir)
        # Revert the parameters to the original values
        self.m.set_state(tmp_params)
        return {'train': tr_acc, 'test': test_acc}


def evaluate_accuracy(ytrue, ypred):

    """
    :param ytrue: Nx1 array of labels
    :param ypred: NxK array of predicted probabilities
    :return:


    """
    return np.mean(np.argmax(ypred, 1) == ytrue)


def evaluate_mnlp(ytrue, ypred):
    """
    :param ytrue: Nx1 array of labels. ytrue \in [0, K-1], where K=# classes
    :param ypred: NxK array of predicted probabilities
    :return:
    """
    probs = ypred[range(ypred.shape[0]), ytrue]
    return - np.mean(np.log(probs))


def write_test_predictions(ytrue, ypred, test_acc, test_mnlp, results_dir):
    """
    :param ytrue: Nx1 array of labels. ytrue \in [0, K-1], where K=# classes
    :param ypred: NxK array of predicted probabilities
    :return:
    """
    if results_dir is not None:
        if not os.path.exists(os.path.expanduser(results_dir)):
            print("Results dir does not exist.")
            print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
            os.makedirs(os.path.expanduser(results_dir))
            print(
                "Created results directory: {}".format(os.path.expanduser(results_dir))
            )
        else:
            print("Results directory already exists.")

    label_pred = np.argmax(ypred, 1)
    df = pd.DataFrame({'y_true': ytrue, 'y_pred': label_pred})
    for ind, col in enumerate(ypred.transpose()):
        df['y_pred_{}'.format(ind)] = col

    predictions_filename = os.path.join(os.path.expanduser(results_dir),  "predictions.csv")
    df.to_csv(predictions_filename, index=None)

    perf_filename = os.path.join(os.path.expanduser(results_dir), "results.csv")
    header = "accuracy_test,  mnlp_test"

    try:
        fh_results = open(perf_filename, "w", buffering=1)
        fh_results.write(header + "\n")
        results_str = "{:04f}, {:04f}\n".format(test_acc, test_mnlp)
        fh_results.write(results_str)

    except IOError:
        print("Could not open results file {}".format(perf_filename))
        return 0  # probably should return something other than success!

    return


# def get_results_handler(results_dir, header, params):
#
#     # setup writing results to disk
#     fh_results = None
#     if results_dir is not None:
#         if not os.path.exists(os.path.expanduser(results_dir)):
#             print("Results dir does not exist.")
#             print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
#             os.makedirs(os.path.expanduser(results_dir))
#             print(
#                 "Created results directory: {}".format(os.path.expanduser(results_dir))
#             )
#         else:
#             print("Results directory already exists.")
#
#         # write parameters file
#         params_filename = os.path.join(os.path.expanduser(results_dir),  "params.csv")
#         try:
#             with open(params_filename, "w", buffering=1) as fh_params:
#                 w = csv.DictWriter(fh_params, params.keys())
#                 w.writeheader()
#                 w.writerow(params)
#
#         except IOError:
#             print("Could not open results file {}".format(params_filename))
#
#         # write headers on results file
#         results_filename = os.path.join(
#             os.path.expanduser(results_dir),  "results.csv"
#         )
#
#         try:
#             fh_results = open(results_filename, "w", buffering=1)
#             # write the column names
#             fh_results.write(
#                 header + "\n"
#             )
#         except IOError:
#             print("Could not open results file {}".format(results_filename))
#             return 0  # probably should return something other than success!
#     return fh_results


def save_parameters(params, results_dir):
    if results_dir is not None:
        if not os.path.exists(os.path.expanduser(results_dir)):
            print("Results dir does not exist.")
            print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
            os.makedirs(os.path.expanduser(results_dir))
            print(
                "Created results directory: {}".format(os.path.expanduser(results_dir))
            )
        else:
            print("Results directory already exists.")

        # write parameters file
        params_filename = os.path.join(os.path.expanduser(results_dir),  "params.csv")
        try:
            with open(params_filename, "w", buffering=1) as fh_params:
                w = csv.DictWriter(fh_params, params.keys())
                w.writeheader()
                w.writerow(params)

        except IOError:
            print("Could not open results file {}".format(params_filename))
    return


@click.command()
@click.option(
    "--dataset",
    default="cora",
    type=click.STRING,
    help="data set name [cora|citeseer|pubmed].",
)
@click.option(
    "--epochs",
    default=10,
    type=click.INT,
    help="Number of epochs [int].",
)
@click.option(
    "--adjacency",
    type=click.STRING,
    help="name of adjacency matrix file [string]",
)
@click.option(
    "--random-seed-np",
    type=click.INT,
    help="global numpy random seed [integer]",
)
@click.option(
    "--random-seed-tf",
    type=click.INT,
    help="global tensorflow random seed [integer]",
)
@click.option(
    "--random-split/--fixed-split",
    default=False,
    help="Use random split (true) or fixed split (false)",
)
@click.option(
    "--split-sizes",
    default=[0.9, 0.75],
    nargs=2,
    type=click.FLOAT,
    help="size of random splits",
)
@click.option(
    "--random-split-seed",
    type=click.INT,
    default=1,
    help="random split seed [integer]",
)
@click.option(
    "--add-val/--no-add-val",
    default=False,
    help="Add 50% of validation for training (true) or not",
)
@click.option(
    "--add-val-seed",
    type=click.INT,
    help="Seed for including validation data in training [integer]",
)
@click.option(
    "--results-dir",
    type=click.STRING,
    help="name of results directory [string]",
)
def main(dataset,
         epochs,
         adjacency,
         random_seed_np,
         random_seed_tf,
         random_split,
         split_sizes,
         random_split_seed,
         add_val,
         add_val_seed,
         results_dir):

    params = click.get_current_context().params
    save_parameters(params, results_dir)

    exp_obj = SSLExperimentNoisy(data_name=dataset,
                                 adj_name=adjacency,
                                 random_seed_np=random_seed_np,
                                 random_seed_tf=random_seed_tf,
                                 random_split=random_split,
                                 split_sizes=split_sizes,
                                 random_split_seed=random_split_seed,
                                 add_val=add_val,
                                 add_val_seed=add_val_seed,
                                 p=0.5,
                                 results_dir=results_dir)

    exp_obj.train(maxiter=epochs, check_obj_every_n_iter=200)

    exp_obj.evaluate(results_dir)


if __name__ == "__main__":
    exit(main())  # pragma: no cover