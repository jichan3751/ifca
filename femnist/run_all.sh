#!/bin/bash
resume_dir="models/outputs/fedavg_pretrained"
resume="models/outputs/fedavg_pretrained/checkpoint_c3_e1_lr0.004_sd0.pkl"
echo "Generating data and pretrain"
pushd fedavg_pretrain
bash generate_data.sh
bash run_fedavg_pretrain.sh
popd

pushd ifca
mkdir -p $resume_dir
cp "../fedavg_pretrain/${resume}" "${resume}"
ln -s "../fedavg_pretrain/data"
echo "Running IFCA and global model (Fedavg) case"
bash run_ifca.sh
popd

pushd local
mkdir -p $resume_dir
cp "../fedavg_pretrain/${resume}" "${resume}"
ln -s "../fedavg_pretrain/data"
echo "Running local model case"
bash run_local.sh
popd

pushd oneshot
mkdir -p $resume_dir
cp "../fedavg_pretrain/${resume}" "${resume}"
ln -s "../fedavg_pretrain/data"
echo "Running one-shot case"
bash run_oneshot.sh
popd
