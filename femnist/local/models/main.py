"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import copy
import random
import time
import pickle
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def main():

    args = parse_args()
    print("ARGS")
    print(args)

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    print("IFCA local")

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()

    client_model = ClientModel(args.seed, *model_params)
    ## IFCA
    ## IFCA end

    # Create server

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    num_clients = len(clients)

    server = Server(client_model, num_clients)

    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    server.set_client_model_indices(clients)

    if args.resume:
        print("---resume all models from {} model zero..".format(args.resume))
        if os.path.exists(args.resume):
            ckpt = pickle.load( open( args.resume, "rb" ) )

            # inject the first model weights, but keep the last 4 wieghts (dense layer w b w b)
            for g_i in range(num_clients):
                for i , weight in enumerate(server.models[g_i]):
                    if i < len(server.models[g_i]) - 2:
                        server.models[g_i][i] = copy.deepcopy(ckpt['params'][0][i])
                    else:
                        continue
        else:
            print("--- {} not found!".format(args.resume))

    if args.checkpoint:
        print("---resume checkpoint from {}...".format(args.checkpoint))
        if os.path.exists(args.checkpoint):
            ckpt = pickle.load( open( args.checkpoint, "rb" ) )
            # import ipdb; ipdb.set_trace()

            # inject the first model weights, but keep the last 4 wieghts (dense layer w b w b)
            for g_i in range(num_clients):
                for i , weight in enumerate(server.models[g_i]):
                    server.models[g_i][i] = copy.deepcopy(ckpt['params'][g_i][i])
        else:
            print("--- {} not found!".format(args.checkpoint))

    stats = []

    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    current_stats = print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
    current_stats['round'] = -1
    stats.append(current_stats)



    # import ipdb; ipdb.set_trace()

    # Simulate training
    for i in range(num_rounds):
        t0 = time.time()

        # checking if norms of each weight same
        # for g_i in range(args.num_groups):
        #     print("DEBUG m{} 0 {:.3f} 2 {:.3f} -5 {:.3f} -4 {:.3f} -2 {:.3f}".format(g_i, np.linalg.norm(server.models[g_i][0]), np.linalg.norm(server.models[g_i][2]), np.linalg.norm(server.models[g_i][-5]), np.linalg.norm(server.models[g_i][-4]), np.linalg.norm(server.models[g_i][-2])))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        # sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        t1 = time.time()

        # Update server model
        server.update_model()
        t2 = time.time()

        print('--- Round %d of %d: Trained %d Clients took t %.3f u %.3f sec ---' % (i + 1, num_rounds, clients_per_round, t1-t0, t2-t1))

        # import ipdb; ipdb.set_trace()
        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            current_stats = print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
            current_stats['round'] = i
            stats.append(current_stats)

    # import ipdb; ipdb.set_trace()
    # Save server model
    # ckpt_path = os.path.join('checkpoints', args.dataset)
    # if not os.path.exists(ckpt_path):
    #     os.makedirs(ckpt_path)
    # save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    # print('Model saved in path: %s' % save_path)


    ckpt = {
        "params": server.models,
        "stats": stats
    }

    best_accuracy = np.max([st['test']['accuracy'] for st in stats])
    print("Best test accuracy : {}".format(best_accuracy))

    # print("saving results to", args.save)
    # os.makedirs(os.path.dirname(args.save), exist_ok=True)
    # pickle.dump( ckpt, open( args.save, "wb" ) )

    # import ipdb; ipdb.set_trace()

    # Close models
    server.close_model()

def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):

    t0 = time.time()
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    t1 = time.time()

    # group_assignment = [0 for _ in range(server.num_groups)]
    # for c_id in train_stat_metrics:
    #     g_i = train_stat_metrics[c_id]['group_index']
    #     group_assignment[g_i] += 1
    #     del train_stat_metrics[c_id]['group_index']

    print("RRound {} train_group".format(num_round), "infer took {:.3f} sec".format(t1-t0))


    ret_train = print_metrics(train_stat_metrics, num_samples, prefix='train_',ct_round= num_round)
    # writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    t0 = time.time()
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    t1 = time.time()

    # group_assignment = [0 for _ in range(server.num_groups)]
    # for c_id in test_stat_metrics:
    #     g_i = test_stat_metrics[c_id]['group_index']
    #     group_assignment[g_i] += 1
    #     del test_stat_metrics[c_id]['group_index']

    print("RRound {} test_group".format(num_round), "infer took {:.3f} sec".format(t1-t0))

    ret_test = print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set),ct_round= num_round)
    # writer(num_round, test_stat_metrics, eval_set)

    return {"train":ret_train, "test":ret_test}


def print_metrics(metrics, weights, prefix='', ct_round = ""):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)

    to_ret = {}

    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print("RRound {} ".format(ct_round), end="")
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))
        to_ret[metric] = np.average(ordered_metric, weights=ordered_weights)

    return to_ret

if __name__ == '__main__':
    main()
