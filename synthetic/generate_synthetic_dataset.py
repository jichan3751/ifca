import argparse
import json
import os
import time

import torch
import numpy as np

from util import *


parser = argparse.ArgumentParser()
parser.add_argument("--project-dir",type=str,default="output")
parser.add_argument("--data-seed",type=int,default=0)
parser.add_argument("--config-override",type=str,default="")
args = parser.parse_args()


def main():
    config = get_config()
    print("config:",config)

    exp = DatasetGenerate(config)
    exp.setup()
    dataset = exp.generate_dataset()
    exp.save()

    # exp.check_dataset() # for debugging
    pass

def get_config():
    # read config json and update the sysarg
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    args_dict = vars(args)
    config.update(args_dict)

    if config["config_override"] == "":
        del config['config_override']
    else:
        # print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config


class DatasetGenerate(object):
    def __init__(self, config, seed = 0):
        self.seed = config['data_seed']
        self.config = config

        assert self.config['m'] % self.config['p'] == 0

    def setup(self):
        # print('seeding', self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        self.dataset_dir = os.path.join(self.config['project_dir'])
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.dataset_fname = os.path.join(self.dataset_dir, 'dataset.pth')

        # param settings
        # p even -> loc: [-3, -1, 1, 3] p = 4
        # p odd -> loc: [-6, -4, -2, 0, 2, 4, 6] p = 7
        p = int(self.config['p'])
        self.param_settings = [ -p + 1 + 2*i for i in range(p)]


    def generate_dataset(self):
        p = self.config['p']
        d = self.config['d']
        m = self.config['m']
        n = self.config['n']


        dataset = {}
        dataset['config'] = self.config

        # generate parameter set for each cluster
        params = []
        for p_i in range(p):
            loc = self.param_settings[p_i]

            param = torch.tensor(np.random.binomial(1, 0.5, size=(d)).astype(np.float32)) * self.config['r']

            params.append(param)

        dataset['params'] = params
        dataset['data'] = []

        # generate dataset for each machine
        cluster_assignment = [m_i//(m//p) for m_i in range(m)] # ex: [0,0,0,0, 1,1,1,1, 2,2,2,2] for m = 12, p = 3
        dataset['cluster_assignment'] = cluster_assignment

        for m_i in range(m):
            p_i = cluster_assignment[m_i]

            data_X = random_normal_tensor(size=(n,d))
            data_y = data_X @ params[p_i]

            noise_y = random_normal_tensor(size=(n), scale = self.config['noise_scale'] )
            data_y = data_y + noise_y

            dataset['data'].append((data_X, data_y))

        self.dataset = dataset
        return dataset

    def save(self):
        torch.save(self.dataset, self.dataset_fname)

        from pathlib import Path
        Path(os.path.join(self.config['project_dir'],'result_data.txt')).touch()

    def check_dataset(self):
        dataset = torch.load(self.dataset_fname)



if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---generate_synthetic_cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))