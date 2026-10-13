#!/bin/bash
# Build the SEDOO lookup table on the CNES cluster.
#
#   mkdir -p sbatch_reports          # once: sbatch resolves --output at submit
#   for g in 1 2 3 4; do sbatch generate/sedoo/build_psf_lut.slurm $g; done
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

GROUP=${1:?usage: sbatch generate/sedoo/build_psf_lut.slurm <1-4>}

case "$GROUP" in
1) SPECIES=(ammonium blackcar) ;;
2) SPECIES=(dust nitrate) ;;
3) SPECIES=(organicm seasalt) ;;
4) SPECIES=(secondary_organic sulphate) ;;
*)
    echo "group must be 1, 2, 3 or 4, got '$GROUP'" >&2
    exit 2
    ;;
esac

SCRATCH=/work/scratch/data/walcark
REPO=/work/scratch/env/walcark/adjeff-article-rse
TRAINING=gauss

export PATH="$SCRATCH/pixi/bin:$PATH"
export ADJEFF_ARTICLE_CACHE="$SCRATCH/adjeff-article/cache"

OUT="$SCRATCH/adjeff-article/sedoo"

cd "$REPO"
mkdir -p sbatch_reports "$ADJEFF_ARTICLE_CACHE" "$OUT"

if [ ! -f .pixi-activate.sh ]; then
    echo "missing .pixi-activate.sh, run on the login node:" >&2
    echo "  pixi install && pixi shell-hook -e default > .pixi-activate.sh" >&2
    exit 1
fi
# `set -u` off around it: the conda activation scripts the environment
# ships read unset variables of their own, and MKL's dies on it.
set +u
source .pixi-activate.sh
set -u

export SMARTG_DIR_AUXDATA="$REPO/inputs/smartg_auxdata"
for needed in acs aerosols atmospheres; do
    if [ ! -d "${SMARTG_DIR_AUXDATA:-}/$needed" ]; then
        echo "missing '$needed' under SMARTG_DIR_AUXDATA='${SMARTG_DIR_AUXDATA:-}'." >&2
        echo "Run on the login node: pixi run download-auxdata" >&2
        exit 1
    fi
done

for species in "${SPECIES[@]}"; do
    echo "=== group $GROUP, $species" >&2
    python generate/sedoo/build_psf_lut.py \
        --species "$species" \
        --training "$TRAINING" \
        --out "$OUT"
done
