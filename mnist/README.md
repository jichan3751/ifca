# IFCA - MNIST

Implementation of IFCA, experiments with MNIST.

## Requirements
* Python3
* Pytorch
* Numpy

## Config.json
```json
    "m" : 2400,            # number of machines (used for splitting the train dataset and setting the number of parallel worker machines)
    "m_test" : 400,        # number of machines (used for splitting the test dataset)
    "p" : 4,               # number of cluster groups (expects m % p == 0 and m_test % p == 0)
    "n" : 100,             # number of images for each ma
    "h1": 200,             # hidden layer size for the NN 
    "num_epochs": 300,     # number of data
    "batch_size":100,      # batch size of local update
    "tau":10,              # number of local epochs in worker machines
    "lr":0.1,              # learning rate
    "data_seed":0,         # random seed for generating data
    "train_seed":0         # random seed for weight initiailization and training

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
The script loops over experiment code with different random seed with 5(=max_procs) processes of experiments running concurrently. After all experiments are done, average test accuracy for each case are printed.

## Notes
* GPU is not used. Only CPU is used (we recommend using cpuonly version of pytorch) 
* Also, it takes several days to run all the experiment. We recommend running this script in cluster with many CPU cores, with max_procs tuned to (num_cpu_cores / 4).