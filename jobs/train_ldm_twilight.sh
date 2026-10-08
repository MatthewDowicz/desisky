#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J desisky-ldm-twi
#SBATCH -o logs/ldm_twilight_%j.out
#SBATCH -e logs/ldm_twilight_%j.err

module load conda
conda activate $PSCRATCH/envs/desisky
export DESISKY_CACHE_DIR=$PSCRATCH/desisky_cache
cd $PSCRATCH/desisky

# InfoVAE trained on the loa extraction (job 59517991, wandb run laced-dew-35)
VAE_CHECKPOINT="models/laced-dew-35.eqx"

XLA_FLAGS="--xla_gpu_autotune_level=0" \
    srun desisky-train-ldm --variant twilight --epochs 5250 \
    --learning-rate 2e-5 --batch-size 256 --dropout-p 0.0 \
    --hidden 32 --levels 2 --emb-dim 32 \
    --vae-path $VAE_CHECKPOINT \
    --data-path training_data/metadata_clean.csv \
    --flux-path training_data/flux_clean.npy \
    --save-dir models/ --wandb --wandb-project desisky-ldm
