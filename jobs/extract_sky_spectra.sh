#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -J desisky-extract
#SBATCH -o logs/extract_%j.out
#SBATCH -e logs/extract_%j.err

# Usage:  sbatch jobs/extract_sky_spectra.sh [release] [nproc]
#         sbatch jobs/extract_sky_spectra.sh loa 32
#
# Extracts <release> sky spectra, one worker per month, into skydata_<release>/.
# Runs on a CPU compute node under SLURM, detached from your login session,
# so it survives logouts and laptop sleep.  Monitor with `squeue --me` and
# logs/extract_<jobid>.out.  Resumable: finished months are skipped, so just
# resubmit if it hits the time limit.
# Needs the DESI software stack (desispec), not the desisky conda env.
RELEASE=${1:-loa}
NPROC=${2:-32}

source /global/common/software/desi/desi_environment.sh main
cd $PSCRATCH/desisky
echo "release=$RELEASE nproc=$NPROC node=$(hostname) started=$(date)"

python scripts/extract_sky_spectra.py --release "$RELEASE" --nproc "$NPROC"
