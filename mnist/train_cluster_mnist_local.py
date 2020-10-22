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

    exp = TrainMNISTClusterLocal(config)
    exp.setup()
    exp.run()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir",type=str,default="output")
    parser.add_argument("--dataset-dir",type=str,default="output")
    # parser.add_argument("--num-epochs",type=float,default=)
    parser.add_argument("--lr",type=float,default=0.1)
    parser.add_argument("--num-epochs",type=int,default=10)
    parser.add_argument("--train-seed",type=int,default=0)
    parser.add_argument("--config-override",type=str,default="")
    parser.add_argument("--checkpoint",type=int,default=0)
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

class TrainMNISTClusterLocal(TrainMNISTCluster):
    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok = True)

        if self.config['checkpoint']:
            self.result_fname = os.path.join(self.config['project_dir'], 'results_ckpt_local.pickle')
        else:
            self.result_fname = os.path.join(self.config['project_dir'], 'results_local.pickle')

        self.load_checkpoint_fname = os.path.join(self.config['project_dir'], 'checkpoint_single.pt')

        self.setup_datasets()
        self.setup_models()

        self.epoch = None
        self.lr = None

    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        torch.manual_seed(self.config['train_seed'])

        m = self.config['m']

        self.models = [ SimpleLinear(h1 = self.config['h1']) for m_i in range(m)] # m models

        self.criterion = torch.nn.CrossEntropyLoss()

        if self.config['checkpoint']:
            print('loading checkpoint from',self.load_checkpoint_fname )

            states = torch.load(self.load_checkpoint_fname)

            # load from first model
            state_dict = states['models'][0]
            for model in self.models:
                model.load_state_dict(state_dict)

        # import ipdb; ipdb.set_trace()


    def run(self):
        num_epochs = self.config['num_epochs']
        lr = self.config['lr']

        results = []

        # epoch -1
        self.epoch = -1

        result = {}
        result['epoch'] = -1

        t0 = time.time()
        res = self.test(train=True)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['train'] = res

        self.print_epoch_stats(res)

        t0 = time.time()
        res = self.test(train=False)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['test'] = res
        self.print_epoch_stats(res)
        results.append(result)

        # this will be used in next epoch
        cluster_assign = result['train']['cluster_assign']

        for epoch in range(num_epochs):
            self.epoch = epoch

            result = {}
            result['epoch'] = epoch

            lr = self.lr_schedule(epoch)
            result['lr'] = lr

            t0 = time.time()
            result['train'] = self.train(cluster_assign, lr = lr)
            t1 = time.time()
            train_time = t1-t0

            t0 = time.time()
            res = self.test(train=True)
            t1 = time.time()
            res['infer_time'] = t1-t0
            res['train_time'] = train_time
            res['lr'] = lr
            result['train'] = res

            self.print_epoch_stats(res)

            # if epoch % 2 == 0 and epoch !=0:
            if 1:

                t0 = time.time()
                res = self.test(train=False)
                t1 = time.time()
                res['infer_time'] = t1-t0
                result['test'] = res
                self.print_epoch_stats(res)

            results.append(result)

            # this will be used in next epoch's gradient update
            cluster_assign = result['train']['cluster_assign']

            if epoch == num_epochs - 1 :
                with open(self.result_fname, 'wb') as outfile:
                    pickle.dump(results, outfile)
                    print(f'result written at {self.result_fname}')


    def get_inference_stats(self, train = True):
        VERBOSE=1

        cfg = self.config
        if train:
            m_test = cfg['m']
            m = cfg['m']

            num_data = 0
            losses = []
            corrects = []
            for m_i in range(m):
                (X, y) = self.load_data(m_i, train=train) # load batch data rotated
                y_logit = self.models[m_i](X)
                loss = self.criterion(y_logit, y) # loss of
                n_correct = self.n_correct(y_logit, y)

                losses.append(loss.item())
                corrects.append( n_correct)

                num_data += X.shape[0]

            loss = np.mean(losses)
            acc = np.sum(corrects) / num_data



        else: # test
            m_test = cfg['m_test']
            m = cfg['m']

            # for each model, get accuracy for all same cluster model

            if N_PROCS > 1:
                losses, accs = run_inference_multiprocess(cfg, self.dataset, self.criterion ,self.models, train)
            else:
                accs = []
                losses = []
                for m_i in range(m):
                    if VERBOSE and m_i % 1 == 0: print(f'm {m_i}/{m} processing \r', end ='')
                    model = self.models[m_i]
                    model_p = self.dataset['train']['cluster_assign'][m_i]

                    (avg_loss, acc) = loss_correct(cfg, self.dataset, self.criterion ,model, model_p, train)

                    accs.append(acc)
                    losses.append(avg_loss)


            # get loss, acc averaged over models
            loss = np.mean(losses)
            acc = np.mean(accs)


        cluster_assign = []
        # for m_i in range(m):
        #     cluster_assign.append(0)

        # check cluster assignment acc
        cl_acc = 0
        cl_ct = []

        res = {} # results
        res['cluster_assign'] = cluster_assign # dummy
        res['loss'] = loss
        res['acc'] = acc
        res['cl_acc'] = cl_acc # dummy
        res['cl_ct'] = cl_ct # dummy
        res['is_train'] = train

        # import ipdb; ipdb.set_trace()

        return res


    def train(self, cluster_assign, lr):
        # cluster_assign is dummy
        VERBOSE = 0

        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        tau = cfg['tau']

        # run local update
        t0 = time.time()

        for m_i in range(m):
            if VERBOSE and m_i % 100 == 0: print(f'm {m_i}/{m} processing \r', end ='')

            (X, y) = self.load_data(m_i)

            model = self.models[m_i]

            for step_i in range(tau):

                y_logit = model(X)
                loss = self.criterion(y_logit, y)

                model.zero_grad()
                loss.backward()
                self.local_param_update(model, lr)

            model.zero_grad()

        t1 = time.time()
        if VERBOSE: print(f'local update {t1-t0:.3f}sec')




if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster single Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))