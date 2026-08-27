#!/bin/bash

#SBATCH --job-name=Hist2ST-B-train
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=dingzetao@sjtu.edu.cn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Train Hist2ST-B (Tumor_EGFR + Mac_EREG) — Hist2ST-closest (fp32, bake=5).
# Needs full a800/A100 GPU (not debuga100 MIG).
# Prerequisite: 01_prepare_visium.py
#
#   cd .../benchmark/beacon_vs_hist2st && sbatch slurm_hist2st_train.slurm

set -euo pipefail

module load gcc cuda

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/script/spatial_transcriptome/pathospatial_model
WORKDIR=${ROOT}/benchmark/beacon_vs_hist2st
CFG=${ROOT}/benchmark/configs/benchmark_hist2st_b.yaml
PYTHON=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/.conda/envs/mambaenv/envs/trident/bin/python

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "config=${CFG}"
nvidia-smi -L || true
"${PYTHON}" -u 02_train.py --config "${CFG}"
