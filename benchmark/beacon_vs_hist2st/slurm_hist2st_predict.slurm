#!/bin/bash

#SBATCH --job-name=Hist2ST-B-pred
#SBATCH --partition=debuga100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=dingzetao@sjtu.edu.cn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Predict all 8 slides with trained Hist2ST-B checkpoints.
# Prerequisite: slurm_hist2st_train.slurm finished.
#
#   cd .../benchmark/beacon_vs_hist2st && sbatch slurm_hist2st_predict.slurm

set -euo pipefail

module load gcc cuda

ROOT=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/script/spatial_transcriptome/pathospatial_model
WORKDIR=${ROOT}/benchmark/beacon_vs_hist2st
CFG=${ROOT}/benchmark/configs/benchmark_hist2st_b.yaml
PYTHON=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/.conda/envs/mambaenv/envs/trident/bin/python

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "config=${CFG}"
nvidia-smi -L || true
"${PYTHON}" -u 03_predict.py --config "${CFG}"
