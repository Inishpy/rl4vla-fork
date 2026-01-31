import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class OpenVLAPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.float32)
        self.action_scale = 1.0

        # openvla: register
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        # self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.base = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )

        # Load LoRA adaptor as a second model for linear combination
        self.vla_lora1 = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla_lora1 = PeftModel.from_pretrained(
            self.vla_lora1,
            "/data/home/co/coimd/rl4vla-fork/wandb/offline-run-20251122_142616-5qr1fw06/glob/steps_0239",    #PutEggplantInBasketScene-v1
            is_trainable=True
        )

        # Linear combination weight for LoRA adaptor (trainable)
        # Remove global lora_weight, will use per-layer beta instead

        # Load two more LoRA adaptors from dummy paths
        self.vla_lora2 = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla_lora2 = PeftModel.from_pretrained(
            self.vla_lora2,
            "/data/home/co/coimd/rl4vla-fork/wandb/offline-run-20251122_142616-upbt77d0/glob/steps_0199",   #PutCarrotOnPlateInScene-v1
            is_trainable=True
        )
        self.vla_lora3 = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla_lora3 = PeftModel.from_pretrained(
            self.vla_lora3,
            "/data/home/co/coimd/rl4vla-fork/wandb/offline-run-20251122_142616-us074evv/glob/steps_0239",  #PutSpoonOnTableClothInScene-v1
            is_trainable=True
        )
        
        #/home/lunet/coimd/RL4VLA/wandb/offline-run-20251122_142616-wldmtz9r/glob/steps_0239  #StackcubesInScene-v1
        #/home/lunet/coimd/RL4VLA/wandb/offline-run-20251122_142616-h53zetbs/glob/steps_0159  #PutOnPlateInScene25VisionImage-v1
        #/home/lunet/coimd/RL4VLA/wandb/offline-run-20251122_142616-b4w5dizr/glob/steps_0159  #PutOnPlateInScene25-v1

        # Patch all LoRA layers to use per-layer trainable betas for 1-3 adaptors
        def patch_lora_layers(base_model, lora1, lora2=None, lora3=None):
            # Build name -> module dicts
            base_mods = dict(base_model.named_modules())
            l1_mods = dict(lora1.named_modules())
            l2_mods = dict(lora2.named_modules()) if lora2 is not None else {}
            l3_mods = dict(lora3.named_modules()) if lora3 is not None else {}

            for name, mod_base in base_mods.items():
                # Only patch layers that exist in base and lora1 (lora1 is required)
                if name not in l1_mods:
                    continue

                mod_l1 = l1_mods[name]
                
                # Check if lora2 and lora3 have this layer (if they're provided)
                mod_l2 = l2_mods.get(name) if lora2 is not None else None
                mod_l3 = l3_mods.get(name) if lora3 is not None else None

                # Heuristic: only patch LoRA-wrapped modules or linear-like modules
                # Adjust the class name check to match your Lora layer class name if needed
                if mod_base.__class__.__name__ != mod_l1.__class__.__name__:
                    continue
                
                # Create beta for base model too
                if not hasattr(mod_base, "beta_base"):
                    device = next(mod_base.parameters()).device
                    num_models = 1 + 1 + (mod_l2 is not None) + (mod_l3 is not None)  # base + lora1 + optional
                    init_value = 1.0 / num_models
                    
                    mod_base.beta_base = torch.nn.Parameter(
                        torch.tensor(init_value, dtype=torch.float32, device=device)
                    )
                    mod_base.register_parameter("beta_base", mod_base.beta_base)
                    
                    # Initialize LoRA betas the same way
                    mod_base.beta1 = torch.nn.Parameter(
                        torch.tensor(init_value, dtype=torch.float32, device=device)
                    )
                    mod_base.register_parameter("beta1", mod_base.beta1)
                    
                    mod_base.beta2 = torch.nn.Parameter(
                        torch.tensor(init_value, dtype=torch.float32, device=device)
                    )
                    mod_base.register_parameter("beta2", mod_base.beta2)
                    
                    mod_base.beta3 = torch.nn.Parameter(
                        torch.tensor(init_value, dtype=torch.float32, device=device)
                    )
                    mod_base.register_parameter("beta3", mod_base.beta3)

                # Save the original forwards so we can call them directly
                base_forward = mod_base.__class__.forward.__get__(mod_base, mod_base.__class__)
                l1_forward = mod_l1.__class__.forward.__get__(mod_l1, mod_l1.__class__)
                l2_forward = mod_l2.__class__.forward.__get__(mod_l2, mod_l2.__class__) if mod_l2 is not None else None
                l3_forward = mod_l3.__class__.forward.__get__(mod_l3, mod_l3.__class__) if mod_l3 is not None else None

                def make_new_forward(base_f, f1, f2, f3):
                    # In the forward pass:
                    def new_forward(self, x, *args, **kwargs):
                        out_base = base_f(x, *args, **kwargs)
                        out_l1 = f1(x, *args, **kwargs)
                        
                        # All models treated equally
                        b_base = torch.clamp(self.beta_base, 0.0, 1.0)
                        b1 = torch.clamp(self.beta1, 0.0, 1.0)
                        
                        result = b_base.to(out_base.dtype) * out_base + b1.to(out_base.dtype) * out_l1
                        
                        if f2 is not None:
                            out_l2 = f2(x, *args, **kwargs)
                            b2 = torch.clamp(self.beta2, 0.0, 1.0)
                            result = result + b2.to(out_base.dtype) * out_l2
                        
                        if f3 is not None:
                            out_l3 = f3(x, *args, **kwargs)
                            b3 = torch.clamp(self.beta3, 0.0, 1.0)
                            result = result + b3.to(out_base.dtype) * out_l3
                        
                        return result
                    return new_forward

                # Bind the new forward to the base module instance
                mod_base.forward = make_new_forward(base_forward, l1_forward, l2_forward, l3_forward).__get__(mod_base, mod_base.__class__)


        patch_lora_layers(self.base, self.vla_lora2, self.vla_lora3)#, self.vla_lora2, self.vla_lora3)
        self.vla = self.base

        # openvla: lora
        if not self.args.vla_load_path:
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", "fc1", "fc2",  # vision
                    "q", "kv", "fc3",  # project
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
                ],
                init_lora_weights="gaussian"
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # set value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        # openvla: optimizer
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        # All per-layer beta parameters are now included via named_parameters
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(**self.tpdv)

        # prompt
        if action is None:
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: "
                           for t in task_description]
        else:
            assert isinstance(action, torch.Tensor)
            # action = action.cpu().numpy() # [B, dim]
            action_str = self.tokenizer.batch_decode(action)

            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                           for t, a in zip(task_description, action_str)]

        inputs = self.processor(task_prompt, images, padding=True)
        inputs = inputs.to(**self.tpdv)

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._preprocess_obs(x)

        values, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )

        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x)

        _, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
        )

        assert len(action.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return action, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        value = self.vla.get_value(**features)

        assert len(value.shape) == 2 and value.shape[1] == 1

        return value

    def get_hidden(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        hs = self.vla.get_hidden(**features)

        return hs

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x, action)

        logprobs, entropy, values = self.vla.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )

        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

        return logprobs, entropy, values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.eval()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")

        json.dump(self.vla.base_model.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        del self.vla
        torch.cuda.empty_cache()

        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location=self.tpdv["device"])
        # No global lora_weight to restore; per-layer betas are handled as parameters

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

class OpenVLAPPO:
    def __init__(self, all_args, policy: OpenVLAPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn

    def train_ppo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)  # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Value loss
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()

        value_loss = value_loss.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
            self.policy.vh_optimizer.step()
            self.policy.vla_optimizer.step()
            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            value_clip_ratio=value_clip_ratio.item(),
            value_old_mean=value_preds.mean().item(),
            values_mean=values.mean().item(),
            returns_mean=returns.mean().item(),
            returns_norm_mean=returns_norm.mean().item(),
            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_grpo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        # value_preds = torch.tensor(value_preds).to(**self.tpdv)
        # returns = torch.tensor(returns).to(**self.tpdv_vn) # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla, self.ppo_grad_norm)
            self.policy.vla_optimizer.step()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_grpo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_grpo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info
