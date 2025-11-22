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

        # --- Beta weights as trainable parameter ---
        # Determine number of environments
        if hasattr(self.args, "all_envs") and self.args.all_envs:
            self.env_list = self.args.all_envs.split(",")
        else:
            self.env_list = [self.args.env_id]
        num_envs = len(self.env_list)
        # Initialize: own mask weight=1, others=0
        beta_init = torch.zeros(num_envs, dtype=torch.float32)
        beta_init[self.args.agent_id] = 1.0
        self.beta_weights = nn.Parameter(beta_init)

        # Track which environments have received LoRA modules
        # True for own env, False for others initially
        self.lora_received_mask = torch.zeros(num_envs, dtype=torch.bool)
        self.lora_received_mask[self.args.agent_id] = True

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
        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )

        # openvla: lora
        # Instead of a single LoRA adapter, create one per environment
        self.lora_adapters = nn.ModuleList()
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
            for _ in range(num_envs):
                # Each adapter is a separate PEFT model on a copy of the base model
                # (Assume get_peft_model returns a model with LoRA injected)
                adapter = get_peft_model(
                    OpenVLAForActionPredictionWithValueHead.from_pretrained(
                        self.args.vla_path,
                        attn_implementation="flash_attention_2",
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map="cuda:" + str(self.device_id),
                        vh_mode="a0",
                    ),
                    lora_config
                )
                self.lora_adapters.append(adapter)
            # The main self.vla remains the base model (without LoRA)
        else:
            # If loading from checkpoint, fallback to original logic for now
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

        # self.vla.print_trainable_parameters()

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

    def forward_with_beta_lora(self, features, *args, **kwargs):
        """
        Forward pass that linearly combines the outputs of all LoRA adapters using beta_weights.
        Assumes self.vla is the base model (without LoRA), and self.lora_adapters is a ModuleList of LoRA models.
        """
        # If no lora_adapters, just use the loaded model directly (single LoRA or base)
        if not hasattr(self, "lora_adapters") or len(self.lora_adapters) == 0:
            return self.vla.predict_action_batch(
                **features,
                unnorm_key=self.args.vla_unnorm_key,
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0),
            )
        # Get base model output
        base_out = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=kwargs.get("do_sample", False),
            temperature=kwargs.get("temperature", 1.0),
        )
        # Each LoRA adapter: run forward, subtract base to get delta, then combine
        deltas = []
        for adapter in self.lora_adapters:
            lora_out = adapter.predict_action_batch(
                **features,
                unnorm_key=self.args.vla_unnorm_key,
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0),
            )
            # Each output is (values, action, logprobs)
            # Subtract base model output to get delta for each output
            lora_delta = tuple(lo - bo for lo, bo in zip(lora_out, base_out))
            deltas.append(lora_delta)
        # Stack and combine deltas using beta_weights
        # Each element in deltas is a tuple (values, action, logprobs)
        # We combine each output type separately

        # Mask out beta logits for environments with no LoRA
        beta_logits = self.beta_weights.clone()
        mask = self.lora_received_mask.to(beta_logits.dtype)
        beta_logits[mask == 0] = float('-inf')
        beta = torch.softmax(beta_logits, dim=0)

        combined = []
        for i in range(len(base_out)):
            stacked = torch.stack([d[i] for d in deltas], dim=0)  # [num_envs, ...]
            beta = beta.to(stacked.device)
            combined_delta = (beta.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0)
            combined.append(base_out[i] + combined_delta)
        return tuple(combined)
    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        # Add beta_weights to params_vla
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        self.params_vla.append(self.beta_weights)
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

        # Use the beta-weighted LoRA combination
        values, action, logprobs = self.forward_with_beta_lora(
            features,
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

        # If no lora_adapters, just use the loaded model directly
        if not hasattr(self, "lora_adapters") or len(self.lora_adapters) == 0:
            value = self.vla.get_value(**features)
            assert len(value.shape) == 2 and value.shape[1] == 1
            return value

        # Use the beta-weighted LoRA combination for value prediction
        base_value = self.vla.get_value(**features)
        deltas = []
        for adapter in self.lora_adapters:
            lora_value = adapter.get_value(**features)
            lora_delta = lora_value - base_value
            deltas.append(lora_delta)
        # Mask out beta logits for environments with no LoRA
        beta_logits = self.beta_weights.clone()
        mask = self.lora_received_mask.to(beta_logits.dtype)
        beta_logits[mask == 0] = float('-inf')
        beta = torch.softmax(beta_logits, dim=0)
        stacked = torch.stack(deltas, dim=0)
        combined_delta = (beta.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0)
        value = base_value + combined_delta

        assert len(value.shape) == 2 and value.shape[1] == 1

        return value

    def get_hidden(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        hs = self.vla.get_hidden(**features)

        return hs

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x, action)

        # If no lora_adapters, just use the loaded model directly
        if not hasattr(self, "lora_adapters") or len(self.lora_adapters) == 0:
            base_out = self.vla.evaluate_action(
                **features,
                unnorm_key=self.args.vla_unnorm_key
            )
            logprobs, entropy, values = base_out
            assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
            assert len(entropy.shape) == 2 and entropy.shape[1] == 1
            assert len(values.shape) == 2 and values.shape[1] == 1
            return logprobs, entropy, values

        # Use the beta-weighted LoRA combination for evaluation
        base_out = self.vla.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )
        deltas = []
        for adapter in self.lora_adapters:
            lora_out = adapter.evaluate_action(
                **features,
                unnorm_key=self.args.vla_unnorm_key
            )
            lora_delta = tuple(lo - bo for lo, bo in zip(lora_out, base_out))
            deltas.append(lora_delta)
        # Mask out beta logits for environments with no LoRA
        beta_logits = self.beta_weights.clone()
        mask = self.lora_received_mask.to(beta_logits.dtype)
        beta_logits[mask == 0] = float('-inf')
        beta = torch.softmax(beta_logits, dim=0)
        combined = []
        for i in range(len(base_out)):
            stacked = torch.stack([d[i] for d in deltas], dim=0)
            beta = beta.to(stacked.device)
            combined_delta = (beta.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0)
            combined.append(base_out[i] + combined_delta)

        logprobs, entropy, values = combined

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

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

    def update_lora_received_mask(self, env_idx):
        """
        Mark that a LoRA module has been received for the given environment index.
        """
        self.lora_received_mask[env_idx] = True

    def consolidate_lora(self):
        """
        Consolidate the current linear combination of LoRA adapters into a single LoRA,
        replace all adapters with this one, and reset the mask so only the current environment is active.
        """
        # Only consolidate if there are multiple adapters and more than one is active
        if self.lora_adapters is None or len(self.lora_adapters) == 0:
            return

        # Use the current beta-masked combination to create a new LoRA adapter
        # For simplicity, take the weighted sum of all LoRA parameters
        with torch.no_grad():
            # Assume all adapters have the same structure
            new_adapter = get_peft_model(
                OpenVLAForActionPredictionWithValueHead.from_pretrained(
                    self.args.vla_path,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="cuda:" + str(self.device_id),
                    vh_mode="a0",
                ),
                LoraConfig(
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
            )
            # Weighted sum of LoRA parameters
            beta_logits = self.beta_weights.clone()
            mask = self.lora_received_mask.to(beta_logits.dtype)
            beta_logits[mask == 0] = float('-inf')
            beta = torch.softmax(beta_logits, dim=0)
            # For each parameter in the LoRA adapter
            for name, param in new_adapter.named_parameters():
                if "lora" in name:
                    # Weighted sum across all adapters
                    stacked = torch.stack([a.state_dict()[name] for a in self.lora_adapters], dim=0)
                    param.data.copy_((beta.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0))
            # Replace all adapters with the new one
            self.lora_adapters = nn.ModuleList([new_adapter])
            # Reset beta_weights and lora_received_mask: only current env is active
            self.beta_weights.data.zero_()
            self.beta_weights.data[0] = 1.0
            self.lora_received_mask.zero_()
            self.lora_received_mask[0] = True

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
