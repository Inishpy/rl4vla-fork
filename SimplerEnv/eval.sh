# conda activate rlvla_env
# cd SimplerEnv

# # Warm-up
# ckpt_path="openvla/openvla-7b"
# unnorm_key="bridge_orig"
# vla_load_path="../openvla/checkpoints/warmup/steps_2000/lora_002000"

# # RL
# ckpt_path="openvla/openvla-7b"
# unnorm_key="bridge_orig"
# vla_load_path="../SimplerEnv/wandb/run-xxx-xxx/glob/steps_xxx" # replace with the actual path

# # SFT
# ckpt_path="../openvla/checkpoints/warmup/steps_2000/merged_002000"
# unnorm_key="sft"
# vla_load_path="../openvla/checkpoints/sft/steps_60000-no_aug/lora_060000"
# RL (pretrained)
ckpt_path="gen-robot/openvla-7b-rlvla-rl"
unnorm_key="bridge_orig"
vla_load_path=""

# start evaluation
for seed in 0 1 2 ; do
    for env_id in \
      "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
      "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \
      "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
      "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
      "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1"
    do
    
      CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
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