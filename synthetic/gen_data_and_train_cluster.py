import argparse
import json
import os
import time
import itertools
import pickle

import torch
import numpy as np

from util import *

from generate_synthetic_dataset import *
from train_cluster import *


parser = argparse.ArgumentParser()
parser.add_argument("--project-dir",type=str,default="output")
parser.add_argument("--dataset-dir",type=str,default="output")
parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--data-seed",type=int,default=0)
parser.add_argument("--train-seed",type=int,default=0)
parser.add_argument("--config-override",type=str,default="")
args = parser.parse_args()

# LR_DECAY = True
LR_DECAY = False

def main():
    config = get_config()
    print("config:",config)

    dataset_generate = DatasetGenerate(config)
    dataset_generate.setup()
    dataset = dataset_generate.generate_dataset()

    exp = TrainCluster(config)
    exp.setup(dataset= dataset)
    exp.run()

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
        print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config




if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))