#!/bin/bash

#SBATCH --job-name=CTransPath
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

# Run from this directory so relative imports / ctrans/ resolve:
#   cd .../benchmark/beacon_vs_path2space && sbatch slurm_CTransPath.slurm

set -euo pipefail

module load gcc cuda

ROOT=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/script/spatial_transcriptome/pathospatial_model
WORKDIR=${ROOT}/benchmark/beacon_vs_path2space
CFG=${ROOT}/benchmark/configs/benchmark_path2space_b.yaml
PYTHON=/dssg/home/acct-drzhihongliu/drzhihongliu-user2/.conda/envs/mambaenv/envs/trident/bin/python

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "config=${CFG}"
"${PYTHON}" -u 01_extract_ctranspath_features.py --config "${CFG}"
