import argparse
import json
import os
import time
import itertools
import pickle
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset


import numpy as np

from train_cluster_mnist import *
from util import *


# LR_DECAY = True
LR_DECAY = False

def main():

    config = get_config()
    print("config:",config)
    config['train_seed'] = config['data_seed']

    exp = TrainMNISTClusterSingle(config)
    exp.setup()
    exp.run()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir",type=str,default="output")
    parser.add_argument("--dataset-dir",type=str,default="output")
    # parser.add_argument("--num-epochs",type=float,default=)
    parser.add_argument("--lr",type=float,default=0.1)
    parser.add_argument("--train-seed",type=int,default=0)
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


class TrainMNISTClusterSingle(TrainMNISTCluster):
    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok = True)

        self.result_fname = os.path.join(self.config['project_dir'], 'results_single.pickle')
        self.checkpoint_fname = os.path.join(self.config['project_dir'], 'checkpoint_single.pt')

        self.setup_datasets()
        self.setup_models()

        self.epoch = None
        self.lr = None



    def get_inference_stats(self, train = True):
        cfg = self.config
        if train:
            m = cfg['m']
            dataset = self.dataset['train']
        else:
            m = cfg['m_test']
            dataset = self.dataset['test']

        p = cfg['p']


        num_data = 0
        losses = {}
        corrects = {}
        for m_i in range(m):
            (X, y) = self.load_data(m_i, train=train) # load batch data rotated
            for p_i in range(p):
                y_logit = self.models[p_i](X)
                loss = self.criterion(y_logit, y) # loss of
                n_correct = self.n_correct(y_logit, y)

                losses[(m_i,p_i)] = loss.item()
                corrects[(m_i,p_i)] = n_correct

            num_data += X.shape[0]

        # calculate loss and cluster the machines
        cluster_assign = []
        for m_i in range(m):
            min_p_i = 0 ########### force this #############
            # machine_losses = [ losses[(m_i,p_i)] for p_i in range(p) ]
            # min_p_i = np.argmin(machine_losses)
            cluster_assign.append(min_p_i)

        # calculate optimal model's loss, acc over all models
        min_corrects = []
        min_losses = []
        for m_i, p_i in enumerate(cluster_assign):

            min_loss = losses[(m_i,p_i)]
            min_losses.append(min_loss)

            min_correct = corrects[(m_i,p_i)]
            min_corrects.append(min_correct)

        loss = np.mean(min_losses)
        acc = np.sum(min_corrects) / num_data


        # check cluster assignment acc
        cl_acc = np.mean(np.array(cluster_assign) == np.array(dataset['cluster_assign']))
        cl_ct = [np.sum(np.array(cluster_assign) == p_i ) for p_i in range(p)]

        res = {} # results
        # res['losses'] = losses
        # res['corrects'] = corrects
        res['cluster_assign'] = cluster_assign
        res['num_data'] = num_data
        res['loss'] = loss
        res['acc'] = acc
        res['cl_acc'] = cl_acc
        res['cl_ct'] = cl_ct
        res['is_train'] = train

        # import ipdb; ipdb.set_trace()

        return res




if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster single Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))