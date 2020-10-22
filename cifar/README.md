# IFCA - CIFAR10

Implementation of IFCA, experiments with CIFAR10.

## Requirements
* Python3
* Tensorflow 1.15
* Numpy

## Config.json
```json
    "m" : 200,            # number of machines (used for splitting the train dataset and setting the number of parallel worker machines)
    "m_test" : 40,        # number of machines (used for splitting the test dataset)
    "p" : 2,               # number of cluster groups (expects m % p == 0 and m_test % p == 0)
    "n" : 500,             # number of images for each ma
    "participation_rate":0.1, # client participation rate
    "num_epochs": 600,     # number of data
    "batch_size":50,      # batch size of local update
    "tau":5,              # number of local epochs in worker machines
    "lr":0.25,              # learning rate
    "data_seed":0,        # random seed for generating data
    "train_seed":0        # random seed for weight initiailization and training

```

## Running the experiments

### To run the single instance of experiment (with data_seed=0 and train_seed=0):
```bash
bash run.sh
```
It will run all the experiments (IFCA, Global Model, Local Model).

### To reproduce results in the paper:
```bash
python run.py   
```
The script loops over experiment code with 5 different random seeds. After all experiments are done, average test accuracy for each case are printed.

## Notes
* The model and data processing codes (cifar10*.py files) are copied from [Tensorflow r0.12 CIFAR10 Tutorial](https://github.com/tensorflow/tensorflow/tree/r0.12/tensorflow/models/image/cifar10)
* GPU is used.
* It takes about a day to run all the experiments.