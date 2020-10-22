#!/bin/bash
resume_dir="models/outputs/fedavg_pretrained"
resume="models/outputs/fedavg_pretrained/checkpoint_c3_e1_lr0.004_sd0.pkl"

# Generating FEMNIST data and train FedAvg to be used as pretrained model
echo "Generating data and pretrain"
pushd fedavg_pretrain
bash generate_data.sh
bash run_fedavg_pretrain.sh
popd

# Run IFCA and Global model case
pushd ifca
mkdir -p $resume_dir
cp "../fedavg_pretrain/${resume}" "${resume}"
ln -s "../fedavg_pretrain/data"
echo "Running IFCA and global model (Fedavg) case"
bash run_ifca.sh
popd

# Run Local model case
pushd local
mkdir -p $resume_dir
cp "../fedavg_pretrain/${resume}" "${resume}"
ln -s "../fedavg_pretrain/data"
echo "Running local model case"
bash run_local.sh
popd

# Run oneshot case
pushd oneshot
mkdir -p $resume_dir
cp "../fedavg_pretrain/${resume}" "${resume}"
ln -s "../fedavg_pretrain/data"
echo "Running one-shot case"
bash run_oneshot.sh
popd
