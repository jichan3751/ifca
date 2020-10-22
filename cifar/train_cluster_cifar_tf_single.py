import argparse
import json
import os
import time
import itertools
import pickle
import copy

import tensorflow as tf

import numpy as np

from util import *
import cifar10

from train_cluster_cifar_tf import *


def main():

    config = get_config()
    config['train_seed'] = config['data_seed']
    print("config:",config)

    exp = TrainCIFARClusterSingle(config)
    exp.setup()
    exp.run()


def get_config():
    arg_seed = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir",type=str,default="output")
    parser.add_argument("--dataset-dir",type=str,default="output")
    # parser.add_argument("--num-epochs",type=float,default=)
    # parser.add_argument("--lr",type=float,default=0.2)
    parser.add_argument("--data-seed",type=int,default=1)
    parser.add_argument("--train-seed",type=int,default=arg_seed)
    parser.add_argument("--config-override",type=str,default="")
    args = parser.parse_args()

    # read config json and update the sysarg
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    args_dict = vars(args)
    config.update(args_dict)

    if config["config_override"] == "":
        del config['config_override']
    else:
        print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config


class TrainCIFARClusterSingle(TrainCIFARCluster):
    def __init__(self, config):
        self.config = config

        assert self.config['m'] % self.config['p'] == 0

    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok = True)

        self.result_fname = os.path.join(self.config['project_dir'], 'results_single')
        self.checkpoint_fname = os.path.join(self.config['project_dir'], 'checkpoint_single')

        set_random_seed(self.config['data_seed'])
        self.setup_datasets()
        self.setup_model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)


        set_random_seed(self.config['data_seed']+self.config['train_seed'])
        self.initialize_models()
        self.initialize_assign_ops()

        # import ipdb; ipdb.set_trace()

        self.epoch = None
        self.lr = None

    def find_good_initializer(self):
        pass

    def test(self, train = True, force_full_nodes = False):

        VERBOSE = 0

        cfg = self.config
        p = cfg['p']
        p_rate = cfg['participation_rate']


        if train:
            m = cfg['m']
            dataset = self.dataset['train']
            if force_full_nodes:
                participating_nodes = list(range(m))
            else:
                participating_nodes = self.participating_nodes
        else:
            m = cfg['m_test']
            dataset = self.dataset['test']
            participating_nodes = list(range(m))

            # DEBUGGING
            # print("DEBUGGING MODEe")
            # participating_nodes = np.random.choice(m, int(m * p_rate), replace = False)


        # get loss and correct from all data


        t_load_model = 0
        t_load_data = 0
        t_infer = 0

        losses = {}
        corrects = {}
        for p_i in range(p):

            tp0= time.time()
            self.put_model_weights(self.model_weights[p_i])
            tp1= time.time()
            t_load_model += tp1-tp0

            for m_i in participating_nodes:

                t00= time.time()
                (X, y) = self.load_node_data(m_i, train=train) # load batch data rotated
                t01= time.time()
                t_load_data += t01-t00

                ti0= time.time()
                (loss, correct) = self.sess.run([self.loss, self.num_correct], feed_dict = {self.x_pl:X, self.y_pl:y})
                ti1= time.time()
                t_infer += ti1-ti0


                losses[(m_i,p_i)] = loss
                corrects[(m_i,p_i)] = correct


        if VERBOSE: print(f"loadmodel {t_load_model:.3f}, load data {t_load_data:.3f}, infer {t_infer:.3f}")


        # calculate loss and cluster the machines
        cluster_assign = [-1 for _ in range(m)]
        for m_i in participating_nodes:
            machine_losses = [ losses[(m_i,p_i)] for p_i in range(p) ]
            min_p_i = np.argmin(machine_losses)
            # cluster_assign[m_i] = min_p_i
            cluster_assign[m_i] = 0 # hack to force single node

        # calculate optimal model's loss, acc over all models

        num_data = len(participating_nodes) * cfg['n']
        min_corrects = []
        min_losses = []
        for m_i in participating_nodes:
            p_i = cluster_assign[m_i]

            min_loss = losses[(m_i,p_i)]
            min_losses.append(min_loss)

            min_correct = corrects[(m_i,p_i)]
            min_corrects.append(min_correct)

        loss = np.mean(min_losses)
        acc = np.sum(min_corrects) / num_data

        # check cluster assignment acc
        # cl_acc = np.mean(np.array(cluster_assign) == np.array(dataset['cluster_assign']))
        cl_ct = [np.sum(np.array(cluster_assign) == p_i ) for p_i in range(p)]


        cluster_assign_ans = dataset['cluster_assign']
        cluster_assign_ans_part = np.array(cluster_assign_ans)[participating_nodes]
        cl_ct_ans = [np.sum(np.array(cluster_assign_ans_part) == p_i ) for p_i in range(p)]

        res = {} # results
        # res['losses'] = losses
        # res['corrects'] = corrects
        # res['cluster_assign'] = cluster_assign
        res['loss'] = loss
        res['acc'] = acc
        # res['cl_acc'] = cl_acc
        res['cl_ct'] = cl_ct
        res['cl_ct_ans'] = cl_ct_ans
        res['is_train'] = train

        if train:
            self.cluster_assign = cluster_assign

        # import ipdb; ipdb.set_trace()

        return res



if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))