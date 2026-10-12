#!/bin/bash
# Build the SEDOO lookup table on the CNES cluster, one job per species.
#
#   mkdir -p sbatch_reports          # once: sbatch resolves --output at submit
#   sbatch generate/sedoo/build_psf_lut.slurm
#   sbatch --array=7 generate/sedoo/build_psf_lut.slurm    # sulphate alone
#
# 1296 fits per species, so a job will be requeued or time out at least
# once.  The script checkpoints slab by slab and skips what is already
# filled, which only works if the cache and the cube outlive the job:
# both go to the scratch, never to the quota'd $HOME, and never to /tmp.
#
#SBATCH --job-name=psf_lut
#SBATCH --export=none
#SBATCH --output=sbatch_reports/psf_lut-%A_%a.out
#SBATCH --error=sbatch_reports/psf_lut-%A_%a.err
#SBATCH --array=0-7
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_std
#SBATCH --qos=gpu_all
#SBATCH -n 16
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=68:00:00
#SBATCH --account=cesbio

set -euo pipefail

SCRATCH=/work/scratch/data/walcark
REPO=/work/scratch/env/walcark/adjeff-article-rse

# `gauss` is what the deposited tables were fitted on; `disks` is what the
# figures of the article use.  A cube built with the other one is not a
# reproduction of the deposit.
TRAINING=gauss

# One species per array task, in the order `SPECIES` declares them.
SPECIES=(
    ammonium
    blackcar
    dust
    nitrate
    organicm
    seasalt
    secondary_organic
    sulphate
)

export PATH="$SCRATCH/pixi/bin:$PATH"
export ADJEFF_ARTICLE_CACHE="$SCRATCH/adjeff-article/cache"
export SMARTG_DIR_AUXDATA="$SCRATCH/smartg/auxdata"

cd "$REPO"
mkdir -p sbatch_reports "$ADJEFF_ARTICLE_CACHE" "$SCRATCH/adjeff-article/sedoo"

pixi run --tls-no-verify -e default python generate/sedoo/build_psf_lut.py \
    --species "${SPECIES[$SLURM_ARRAY_TASK_ID]}" \
    --training "$TRAINING" \
    --out "$SCRATCH/adjeff-article/sedoo"
