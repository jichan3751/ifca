#!/usr/bin/env bash

export OMP_NUM_THREADS=4

mkdir -p output
python -u train_cluster_mnist.py # runs IFCA
python -u train_cluster_single.py # runs Global model case
python -u train_cluster_local.py # runs local model
