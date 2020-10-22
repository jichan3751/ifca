#!/bin/bash
num_rounds="2000"
fedavg_lr="0.004"
clients_per_round="3"
num_epochs="1"
sd="0"

cd models/

mkdir -p outputs
mkdir -p outputs/fedavg_pretrained

python -u main.py \
    -dataset 'femnist' \
    -model 'cnn' \
    --num-rounds ${num_rounds} \
    --clients-per-round ${clients_per_round} \
    --num-epochs ${num_epochs} -lr ${fedavg_lr} \
    --seed $sd \
    --save ./outputs/fedavg_pretrained/checkpoint_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.pkl \
    | tee -i ./outputs/fedavg_pretrained/main_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.txt



# trap 'killall' INT TERM

# killall() {
#     trap '' INT TERM     # ignore INT and TERM while shutting down
#     echo "**** Shutting down... ****"     # added double quotes
#     kill -TERM 0         # fixed order, send TERM not INT
#     wait
#     echo DONE
# }

# clients_per_round="35"
# python -u main.py \
#     -dataset 'femnist' \
#     -model 'cnn' \
#     --num-rounds ${num_rounds} \
#     --clients-per-round ${clients_per_round} \
#     --num-epochs ${num_epochs} -lr ${fedavg_lr} \
#     --num-groups ${num_groups} \
#     --seed $train_seed \
#     --save ./outputs/fedavg_pretrained/checkpoint_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.pkl \
#     > ./outputs/fedavg_pretrained/main_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${train_seed}.txt &

# clients_per_round="3"

# python -u main.py \
#     -dataset 'femnist' \
#     -model 'cnn' \
#     --num-rounds ${num_rounds} \
#     --clients-per-round ${clients_per_round} \
#     --num-epochs ${num_epochs} -lr ${fedavg_lr} \
#     --num-groups ${num_groups} \
#     --seed $train_seed \
#     --save ./outputs/fedavg_pretrained/checkpoint_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${sd}.pkl \
#     > ./outputs/fedavg_pretrained/main_g${num_groups}_c${clients_per_round}_e${num_epochs}_lr${fedavg_lr}_sd${train_seed}.txt &

# wait
