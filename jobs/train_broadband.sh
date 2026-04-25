#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J desisky-broadband
#SBATCH -o logs/broadband_%j.out
#SBATCH -e logs/broadband_%j.err

module load conda
conda activate $PSCRATCH/envs/desisky
export DESISKY_CACHE_DIR=$PSCRATCH/desisky_cache
cd $PSCRATCH/desisky

XLA_FLAGS="--xla_gpu_autotune_level=0" \
    srun desisky-train-broadband --epochs 2000 \
    --data-path training_data/metadata_moon.csv \
    --save-dir models/ --wandb
