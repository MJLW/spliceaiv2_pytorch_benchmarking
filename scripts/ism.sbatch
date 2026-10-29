#!/bin/bash
# SNV saturation mutagenesis of gene haplotypes on one GPU.
#
#   sbatch scripts/ism.sbatch --variants haplotypes.tsv --regions genes.tsv \
#       --fasta GRCh38.fa --out-dir results/ism-run1
#
# All arguments go to spliceai2-ism. Submit from the repository root and
# adjust the #SBATCH lines to your cluster. To split genes over a job array,
# pass --genes and a separate --out-dir per task.
#
# Compute nodes without internet: run `uv sync` and download the weights once
# on a login node (`uv run hf auth login`, then
# `uv run python -c "from spliceai2_ism.weights import download_models; download_models()"`)
# with HF_HOME on a filesystem the compute nodes share.
#SBATCH --job-name=spliceai2-ism
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=spliceai2-ism-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
uv run spliceai2-ism "$@"
