#!/bin/bash

cd models/

num_rounds="4000"
num_groups="3"
fedavg_lr="0.004"
clients_per_round="6"
num_epochs="1"
sd="0"


resume="./outputs/fedavg_pretrained/checkpoint_c3_e1_lr0.004_sd0.pkl" # pretrained model
mkdir -p outputs

num_groups="3"
python -u main.py \
    -dataset 'femnist' \
    -model 'cnn' \
    --num-rounds ${num_rounds} \
    --clients-per-round ${clients_per_round} \
    --num-epochs ${num_epochs} -lr ${fedavg_lr} \
    --num-groups ${num_groups} \
    --seed $sd \
    --resume $resume \
    --save ./outputs/checkpoint_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.pkl \
    |tee -i ./outputs/main_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.txt


num_groups="2"
python -u main.py \
    -dataset 'femnist' \
    -model 'cnn' \
    --num-rounds ${num_rounds} \
    --clients-per-round ${clients_per_round} \
    --num-epochs ${num_epochs} -lr ${fedavg_lr} \
    --num-groups ${num_groups} \
    --seed $sd \
    --resume $resume \
    --save ./outputs/checkpoint_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.pkl \
    |tee -i ./outputs/main_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.txt




