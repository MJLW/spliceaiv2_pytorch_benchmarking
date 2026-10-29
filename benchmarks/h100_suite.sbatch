#!/bin/bash
# Runs the complete SpliceAI2 benchmark suite on one H100.
#
#   sbatch benchmarks/h100_suite.sbatch                 # defaults: 500 predictions each
#   sbatch benchmarks/h100_suite.sbatch --repeats 6     # extra args go to spliceai2-suite
#
# Submit from the repository root. Adjust the #SBATCH lines to your cluster
# (partition, account, GPU type). Compute nodes without internet: download
# the weights once on a login node (see README) and pass --model-dir, or set
# HF_HOME to a shared cache.
#SBATCH --job-name=spliceai2-suite
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=spliceai2-suite-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

nvidia-smi
uv sync
uv run spliceai2-suite --out-dir "results/suite-${SLURM_JOB_ID:-local}" "$@"
