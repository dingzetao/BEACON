#!/bin/bash

#SBATCH --job-name=HisToGene-B-pred
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=dingzetao@sjtu.edu.cn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Predict all 8 slides. Prerequisite: train finished.
#
#   cd .../benchmark/beacon_vs_histogene && sbatch slurm_histogene_predict.slurm

set -euo pipefail

module load gcc cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/script/spatial_transcriptome/pathospatial_model
WORKDIR=${ROOT}/benchmark/beacon_vs_histogene
CFG=${ROOT}/benchmark/configs/benchmark_histogene_b.yaml
PYTHON=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/.conda/envs/mambaenv/envs/trident/bin/python

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "config=${CFG}"
nvidia-smi -L || true
"${PYTHON}" -u 03_predict.py --config "${CFG}"
