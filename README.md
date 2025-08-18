# VLA-RL-Study: What Can RL Bring to VLA Generalization? An Empirical Study

[![arXiv](https://img.shields.io/badge/arXiv-2505.19789-red.svg)](http://arxiv.org/abs/2505.19789)
[![Website](https://img.shields.io/badge/Website-RLVLA-green.svg)](https://rlvla.github.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-models-yellow.svg)](https://huggingface.co/collections/gen-robot/rlvla-684bc48aa6cf28bac37c57a2)

## Introduction

This repository contains the code for the paper [What Can RL Bring to VLA Generalization? An Empirical Study](https://arxiv.org/abs/2505.19789).
The pretrained checkpoints are available at [HuggingFace](https://huggingface.co/collections/gen-robot/rlvla-684bc48aa6cf28bac37c57a2).

## Install

### OpenVLA, Maniskill, Training Pipeline

```bash
# create conda env: rlvla_env
conda create -n rlvla_env -y python=3.10
conda activate rlvla_env

# install dependencies
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install other dependencies
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..

# optional: for ubuntu 2204
# sudo apt-get install libglvnd-dev
```

### RLDS Dataset Maker

Used for building VLA warm-up dataset and OpenVLA SFT datasets.

```bash
# create conda env: rlds_env
cd openvla/rlds_dataset_builder
conda env create -f environment_ubuntu.yml
```

### Octo Inference

Used for collecting data with Octo-Small, when building VLA warm-up dataset.

```bash
conda create -n octo_env -y python=3.10
conda activate octo_env

git clone https://github.com/octo-models/octo.git

cd ManiSkill && pip install -e . && cd ..

cd octo && pip install -e . && pip install -r requirements.txt && cd ..
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "nvidia-cudnn-cu11>=8.7,<9.0" --index-url https://download.pytorch.org/whl/cu118
pip install -U tyro
pip install scipy==1.12.0

cd SimplerEnv && pip install -e . && cd ..
```

## Train

### Warm-up OpenVLA

#### Collect Data with Octo-Small

Collect data with Octo-Small to build the warm-up dataset. Average Octo-Small success rate is about 14% on this task.

```bash
conda activate octo_env
cd SimplerEnv
cuda=0

# for OpenVLA warm-up (extra 5 trajectories for performance evaluation)
CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python simpler_env/eval_ms3_collect.py \
  --env_id "PutCarrotOnPlateInScene-v1"\
  --num-episodes 75 --num-envs 64 --seed 0

# try to increase `num-episodes` if not enough successful trajectories is collected
```

#### Collect Data with motion planner

Collect data with motion planner to build the warm-up dataset and SFT dataset.

```bash
conda activate rlvla_env
cd ManiSkill
cuda=0

# for OpenVLA warm-up (extra 5 trajectories for performance evaluation)
CUDA_VISIBLE_DEVICES=$cuda \
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Single-v1" \
  --save_video --save_data --num_procs 1 --num_traj 75 --seed=0

# for SFT (extra 16 trajectories for performance evaluation)
CUDA_VISIBLE_DEVICES=$cuda \
python -m mani_skill.examples.motionplanning.widowx.collect_simpler \
  -e "PutOnPlateInScene25Main-v3" \
  --save_video --save_data --num_procs 16 --num_traj 16400 --seed=100
```

#### Build VLA Warm-up Dataset

```bash
conda activate rlds_env

cd openvla/rlds_dataset_builder/warmup_dataset
tfds build --overwrite
cd ../../../ # at the root dir of this project
mkdir -p datasets
mv -T ~/tensorflow_datasets/example_dataset datasets/warmup
```

#### Warm-up OpenVLA

```bash
conda activate rlvla_env
cd openvla

# 1. Train LoRA
cuda="0,1,2,3"
task_name="warmup"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 2000 \
  --eval_steps 50 \
  --save_steps "0,500,1000,1500,2000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --unnorm_key="bridge_orig" \
  --wandb_project "RLVLA_sft"

# for 80G GPU, max batch size is 20
# for 40G GPU, max batch size is 8

# 2. Merge LoRA
cuda="0"
task_name="warmup"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/steps_2000" \
  --lora_name "lora_002000"
```

### RL

```bash
conda activate rlvla_env
cd SimplerEnv

#cuda="0,1" # env on GPU-0, model on GPU-1 (for 40G GPU)
cuda="0" # env and model on the same GPU (for 80G GPU)

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="PPO-pc25m_v3-warmup" \
  --env_id="PutOnPlateInScene25Main-v3" \
  --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
  --vla_load_path="../openvla/checkpoints/warmup/steps_2000/lora_002000" \
  --seed=0
```

- GRPO: add `--alg_name="grpo"`
- GRPO (s): add `--alg_name="grpo"` and `--use_same_init`
- PPO from scratch: remove `--vla_load_path` arg

### SFT

#### Build OpenVLA SFT Dataset

```bash
conda activate rlds_env

# ulimit -n 17000 # avoid "too many open files" error

cd openvla/rlds_dataset_builder/sft_dataset
tfds build --overwrite
cd ../../../
mkdir -p datasets
mv -T ~/tensorflow_datasets/example_dataset datasets/sft
```

#### SFT Train

```bash
conda activate rlvla_env
cd openvla

cuda="0,1,2,3"

task_name="sft"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 4 ../openvla/vla-scripts/finetune.py \
  --vla_path "../openvla/checkpoints/warmup/steps_2000/merged_002000" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 60000 \
  --eval_steps 200 \
  --save_steps "0,2500,5000,7500,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project "RLVLA_sft"
```

## Evaluate

### Trained from scratch

```bash
conda activate rlvla_env
cd SimplerEnv

# Warm-up
ckpt_path="openvla/openvla-7b"
unnorm_key="bridge_orig"
vla_load_path="../openvla/checkpoints/warmup/steps_2000/lora_002000"

# RL
ckpt_path="openvla/openvla-7b"
unnorm_key="bridge_orig"
vla_load_path="../SimplerEnv/wandb/run-xxx-xxx/glob/steps_xxx" # replace with the actual path

# SFT
ckpt_path="../openvla/checkpoints/warmup/steps_2000/merged_002000"
unnorm_key="sft"
vla_load_path="../openvla/checkpoints/sft/steps_60000-no_aug/lora_060000"


# start evaluation
for seed in 0 1 2 ; do
    for env_id in 
      "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \ 
      "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \ 
      "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
      "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \ 
      "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1" ; \ 
    do
    
      CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python simpler_env/train_ms3_ppo.py \
        --vla_path="${ckpt_path}" --vla_unnorm_key="${unnorm_key}" \
        --vla_load_path="${vla_load_path}" \
        --env_id="${env_id}" \
        --seed=${seed} \
        --buffer_inferbatch=64 \
        --no_wandb --only_render
    done
done

# for 40G GPU, set `--buffer_inferbatch=16` to avoid OOM
```

### Pre-trained checkpoints

The pretrained checkpoints (warm-upped, RL and SFT) are available at [HuggingFace](https://huggingface.co/collections/gen-robot/rlvla-684bc48aa6cf28bac37c57a2).
Follow the evaluation scripts in the above section, and replace the environment variable with the pretrained checkpoint path.

```bash
# Warm-up (pretrained)
ckpt_path="gen-robot/openvla-7b-rlvla-warmup"
unnorm_key="bridge_orig"
vla_load_path=""

# RL (pretrained)
ckpt_path="gen-robot/openvla-7b-rlvla-rl"
unnorm_key="bridge_orig"
vla_load_path=""

# SFT (pretrained)
ckpt_path="gen-robot/openvla-7b-rlvla-sft_16k"
unnorm_key="sft"
vla_load_path=""
```


### Gather results

1. Option 1: Manually check the results and visualization videos: at `SimplerEnv/wandb/offline-run-xxx-xxx/glob/`
2. Option 2: Calculate statistics: at `SimplerEnv/scripts` run `python calc_statistics.py`, then check the results at `SimplerEnv/scripts/stats`

Task definition:

1. `PutOnPlateInScene25VisionImage-v1`-`test`: unseen table
2. `PutOnPlateInScene25VisionTexture03-v1`-`test`: dynamic texture (weak)
3. `PutOnPlateInScene25VisionTexture05-v1`-`test`: dynamic texture (strong)
4. `PutOnPlateInScene25VisionWhole03-v1`-`test`: dynamic noise (weak)
5. `PutOnPlateInScene25VisionWhole05-v1`-`test`: dynamic noise (strong)
6. `PutOnPlateInScene25Carrot-v1`-`train`: similar to training setting
7. `PutOnPlateInScene25Carrot-v1`-`test`: unseen objects
8. `PutOnPlateInScene25Plate-v1`-`test`: unseen receptacles
9. `PutOnPlateInScene25Instruct-v1`-`test`: unseen instructions
10. `PutOnPlateInScene25MultiCarrot-v1`-`train`: multi-object (both seen)
11. `PutOnPlateInScene25MultiCarrot-v1`-`test`: multi-object (both unseen)
12. `PutOnPlateInScene25MultiPlate-v1`-`train`: distractive receptacle
13. `PutOnPlateInScene25MultiPlate-v1`-`test`: multi-receptacle (both unseen)
14. `PutOnPlateInScene25Position-v1`-`test`: unseen position (object & receptacle)
15. `PutOnPlateInScene25EEPose-v1`-`test`: unseen robot init pose
16. `PutOnPlateInScene25PositionChangeTo-v1`-`test`: mid-episode object reposition
