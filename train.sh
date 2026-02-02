#!/bin/bash
#SBATCH --job-name=merge_sac_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1            # we manage parallelism manually
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --account=Soltoggio2025a
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err


echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

module purge
module load CUDA/12.4.0
source ~/.bashrc
conda activate rlvla_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ALGORITHM="simple"
SEEDS=(0 1 2 3)
ENVS=("PutCarrotOnPlateInScene-v1" "PutSpoonOnTableClothInScene-v1"
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1"
    "PutEggplantInBasketScene-v1")
SCRIPT=/data/home/co/coimd/rl4vla-fork/SimplerEnv/simpler_env/train_ms3_ppo.py

# Launch 8 runs, 2 per GPU
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    ENV_ID="PutEggplantInBasketScene-v1"
    GPU_ID=$(( i % 4 ))

    echo "Launching seed $SEED on GPU $GPU_ID"

    LOGFILE="logs_seed${SEED}_${ALGORITHM}.out"
    {
        echo "Log for seed $SEED started at: $(date)"
        echo "GPU: $GPU_ID"
        echo ""
    } >> "$LOGFILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 $SCRIPT \
        --env_id $ENV_ID \
        --seed $SEED \
        >> "$LOGFILE" 2>&1 &
done


wait  # wait for all 8 background jobs

echo "=========================================="
echo "All runs completed at: $(date)"
echo "=========================================="
