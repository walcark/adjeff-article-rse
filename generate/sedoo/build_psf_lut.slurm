#!/bin/bash
# Build the SEDOO lookup table on the CNES cluster.
#
#   mkdir -p sbatch_reports          # once: sbatch resolves --output at submit
#   sbatch generate/sedoo/build_psf_lut.slurm
#
# 1296 fits per species, so the job will time out before the eight are
# done.  The script checkpoints slab by slab and skips what is already
# filled, so resubmitting it carries on, which only works if the cache
# and the cubes outlive the job: both go to the scratch, never to the
# quota'd $HOME, and never to /tmp.
#
#SBATCH --job-name=psf_lut
#SBATCH --export=none
#SBATCH --output=sbatch_reports/psf_lut-%j.out
#SBATCH --error=sbatch_reports/psf_lut-%j.err
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

export PATH="$SCRATCH/pixi/bin:$PATH"
export ADJEFF_ARTICLE_CACHE="$SCRATCH/adjeff-article/cache"
export SMARTG_DIR_AUXDATA="$SCRATCH/smartg/auxdata"

OUT="$SCRATCH/adjeff-article/sedoo"

cd "$REPO"
mkdir -p sbatch_reports "$ADJEFF_ARTICLE_CACHE" "$OUT"

# One species after another, on the one GPU the job holds.  Comment a line
# out to leave its cube as it stands; the script would skip it anyway once
# it is full.
for species in ammonium blackcar dust nitrate organicm seasalt \
               secondary_organic sulphate; do
    echo "=== $species" >&2
    pixi run --tls-no-verify -e default python generate/sedoo/build_psf_lut.py \
        --species "$species" \
        --training "$TRAINING" \
        --out "$OUT"
done
