#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 256
#SBATCH -J desisky-extract
#SBATCH -o logs/extract_%j.out
#SBATCH -e logs/extract_%j.err

# Usage:  sbatch jobs/extract_sky_spectra.sh [release] [nproc]
#         sbatch jobs/extract_sky_spectra.sh loa 128
#
# Extracts <release> sky spectra into skydata_<release>/ using one full CPU
# node in the debug QOS (starts within minutes; 30 min limit).  The full loa
# set (20,196 exposures) takes ~10-15 min with 128 workers.  Runs under SLURM,
# detached from your login session.  Resumable: finished months are skipped,
# so if it hits the limit just resubmit.  Monitor with `squeue --me` and
# logs/extract_<jobid>.out.
#
# Slower but no time pressure (shared QOS, long queue when the machine is busy):
#   sbatch -q shared -t 8:00:00 -c 64 jobs/extract_sky_spectra.sh loa 32
#
# Needs the DESI software stack (desispec), not the desisky conda env.
RELEASE=${1:-loa}
NPROC=${2:-128}

source /global/common/software/desi/desi_environment.sh main
cd $PSCRATCH/desisky
echo "release=$RELEASE nproc=$NPROC node=$(hostname) started=$(date)"

python scripts/extract_sky_spectra.py --release "$RELEASE" --nproc "$NPROC"
