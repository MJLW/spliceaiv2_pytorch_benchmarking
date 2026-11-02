#!/bin/bash
# Batch-size sweep on one H200: naive full passes vs reference + incremental,
# random and consecutive substitutions, with torch.profiler traces.
#
#   sbatch benchmarks/h200_sweep.sbatch                    # defaults
#   sbatch benchmarks/h200_sweep.sbatch --layouts random   # extra args go to spliceai2-sweep
#
# Submit from the repository root. Adjust the #SBATCH lines to your cluster
# (partition, account, GPU type). The naive axis dominates the run time: each
# run times 500 full forward passes per round. Compute nodes without
# internet: download the weights once on a login node (see README) and pass
# --model-dir, or set HF_HOME to a shared cache.
#SBATCH --job-name=spliceai2-sweep
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=spliceai2-sweep-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

nvidia-smi
# torch.compile's cache on shared storage (the default is node-local /tmp)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.inductor_cache}"
uv sync
uv run spliceai2-sweep --profile --profile-stacks --out-dir "results/sweep-${SLURM_JOB_ID:-local}" "$@"
