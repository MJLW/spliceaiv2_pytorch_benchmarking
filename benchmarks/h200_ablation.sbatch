#!/bin/bash
# Ablation of the incremental optimizations on one H200: one spliceai2-mutbench
# run per configuration and sequence length (see
# benchmarks/spliceai2_bench/ablation.py), each reference + 499 random
# substitutions of 10,000, 20,000, 50,000 and 100,000 positions at variant
# batch 32.
#
#   sbatch benchmarks/h200_ablation.sbatch                          # everything
#   sbatch benchmarks/h200_ablation.sbatch --configs none,all,naive  # extra args go to spliceai2-ablation
#   sbatch benchmarks/h200_ablation.sbatch --seq-lens 100000         # one length
#
# Submit from the repository root and adjust the #SBATCH lines to your
# cluster. About 84 runs (21 configurations x 4 lengths) of 3-10 minutes; the
# first run of each configuration and length pays for torch.compile, later
# jobs reuse .inductor_cache/.
#SBATCH --job-name=spliceai2-ablation
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=spliceai2-ablation-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

nvidia-smi
# torch.compile's cache on shared storage (the default is node-local /tmp)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/.inductor_cache}"
uv sync
uv run spliceai2-ablation --out-dir "results/ablation-${SLURM_JOB_ID:-local}" "$@"
