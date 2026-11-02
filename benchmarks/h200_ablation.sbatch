#!/bin/bash
# Ablation of the incremental optimizations on one H200: one spliceai2-mutbench
# run per configuration (see benchmarks/spliceai2_bench/ablation.py), each
# reference + 499 random substitutions of 100,000 positions at variant batch 32.
#
#   sbatch benchmarks/h200_ablation.sbatch                          # every configuration
#   sbatch benchmarks/h200_ablation.sbatch --configs none,all,naive  # extra args go to spliceai2-ablation
#
# Submit from the repository root and adjust the #SBATCH lines to your
# cluster. About 18 runs of 5-10 minutes; the first run of each configuration
# pays for torch.compile, later jobs reuse .inductor_cache/.
#SBATCH --job-name=spliceai2-ablation
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=spliceai2-ablation-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

nvidia-smi
# torch.compile's cache on shared storage (the default is node-local /tmp)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.inductor_cache}"
uv sync
uv run spliceai2-ablation --out-dir "results/ablation-${SLURM_JOB_ID:-local}" "$@"
