#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J desisky-vae
#SBATCH -o logs/vae_%j.out
#SBATCH -e logs/vae_%j.err

module load conda
conda activate $PSCRATCH/envs/desisky
export DESISKY_CACHE_DIR=$PSCRATCH/desisky_cache
cd $PSCRATCH/desisky

CUDA_VISIBLE_DEVICES=0 XLA_FLAGS="--xla_gpu_autotune_level=0" \
    srun python scripts/nersc_train_vae.py --epochs 6000 \
    --data-path training_data/flux_clean.npz \
    --metadata-path training_data/metadata_clean.csv \
    --save-dir models/ --wandb --viz-every 50
