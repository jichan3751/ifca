#!/bin/bash
mkdir -p out

python -u train_cluster_cifar_tf.py # IFCA
python -u train_cluster_cifar_tf_single.py # Global Model
python -u train_cluster_cifar_tf_local.py # Local Model
