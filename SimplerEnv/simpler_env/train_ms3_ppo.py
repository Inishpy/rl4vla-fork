
import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated, List
import torch
import numpy as np
import tyro
import wandb
import subprocess
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

# Added for logging and Excel
import datetime
import pandas as pd
import contextlib
import sys
import logging
import ot  # For Wasserstein embeddings
import pickle
import multiprocessing as mp

from simpler_env.env.simpler_wrapper import SimlerWrapper
import numpy as np
import random

class FifoReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=100000, episode_len=8, num_envs=16, gamma=0.99, gae_lambda=0.95, buffer_minibatch=2, alg_grpo_fix=True):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # FIFO storage for off-policy
        self.observations = np.zeros((capacity, *obs_dim), dtype=np.uint8)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_dim), dtype=np.uint8)
        self.dones = np.zeros((capacity, 1), dtype=bool)
        self.instruction = ["" for _ in range(capacity)]

        self.ptr = 0
        self.size = 0

        # PPO-style storage for on-policy
        self.episode_len = episode_len
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_minibatch = buffer_minibatch
        self.alg_grpo_fix = alg_grpo_fix

        # On-policy storage for PPO (shaped for batched environments)
        self.value_preds = np.zeros((self.episode_len + 1, self.num_envs, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_len, self.num_envs, 1), dtype=np.float32)
        self.masks = np.ones((self.episode_len + 1, self.num_envs, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_len, self.num_envs, act_dim), dtype=np.float32)
        self.advantages = np.zeros((self.episode_len, self.num_envs, 1), dtype=np.float32)
        self.rewards = np.zeros((self.episode_len, self.num_envs, 1), dtype=np.float32)
        self.actions = np.zeros((self.episode_len, self.num_envs, act_dim), dtype=np.float32)
        self.observations = np.zeros((self.episode_len + 1, self.num_envs, *obs_dim), dtype=np.uint8)
        self.instruction = ["" for _ in range(self.num_envs)]
        self.step = 0
        self.full = False

    def insert(self, obs, action, reward, next_obs, done, instruction=""):
        # FIFO insert
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.instruction[self.ptr] = instruction

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def insert_ppo(self, obs, actions, action_log_probs, value_preds, rewards, masks, instruction=None):
        # On-policy insert for PPO (batched for num_envs)
        # obs: (num_envs, *obs_dim)
        # actions: (num_envs, act_dim)
        # action_log_probs: (num_envs, act_dim)
        # value_preds: (num_envs, 1)
        # rewards: (num_envs, 1)
        # masks: (num_envs, 1)
        # instruction: list of str, len=num_envs

        self.observations[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if instruction is not None:
            self.instruction = instruction
        self.step = (self.step + 1) % self.episode_len
        if self.step == 0:
            self.full = True

    def sample(self, batch_size):
        idxs = random.sample(range(self.size), batch_size)
        batch = dict(
            observations=self.observations[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_observations=self.next_observations[idxs],
            dones=self.dones[idxs],
            instruction=[self.instruction[i] for i in idxs]
        )
        return batch

    def __len__(self):
        return self.size

    def endup(self, next_value):
        self.value_preds[-1] = next_value

    def compute_returns_ppo(self):
        """
        Compute returns and advantages for the most recent episode only.
        This prevents out-of-bounds errors when the buffer is much larger than episode_len.
        """
        episode_len = self.episode_len
        num_envs = self.num_envs
        # Only operate on the last episode_len steps
        rewards = self.rewards
        value_preds = self.value_preds
        masks = self.masks
        returns = np.zeros_like(rewards)
        gae = np.zeros((self.num_envs, 1), dtype=np.float32)
        for step in reversed(range(self.episode_len)):
            vt1 = value_preds[step + 1]
            vt = value_preds[step]
            delta = rewards[step] + self.gamma * vt1 * masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
            returns[step] = gae + vt
        advantages = returns - value_preds[:-1]
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        self.returns = returns

    def compute_returns_grpo(self):
        if self.alg_grpo_fix:
            rewards_valid = self.rewards[self.rewards != 0]
            rewards_norm = self.rewards.copy()
            rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
            rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
        else:
            rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)
        returns = 0
        for step in reversed(range(self.rewards.shape[0])):
            returns = rewards_norm[step] + self.masks[step + 1] * returns
            self.returns[step] = returns
        self.advantages = self.returns.copy()

    def get_minibatch_count(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads
        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch
        return num_mini_batch

    def feed_forward_generator(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads
        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]
        obs = self.observations[:-1].reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = self.advantages.reshape(-1, 1)
        for indices in sampler:
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]
            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)


signal.signal(signal.SIGINT, signal.SIG_DFL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "StackGreenCubeOnYellowCubeBakedTexInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    name: str = "MOSAIC-test"
    num_envs: int = 32
    episode_len: int = 80
    use_same_init: bool = False
    steps_max: int = 200000
    steps_vh: int = 0
    interval_eval: int = 10
    interval_save: int = 15
    buffer_inferbatch: int = 4
    buffer_minibatch: int = 2
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6
    alg_name: str = "ppo"
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0
    wandb: bool = False
    only_render: bool = False
    render_info: bool = False
    num_eval_runs: int = 1
    # MOSAIC-specific args
    comm_interval: int = 5
    agent_id: int = 0
    all_envs: str = ""
    lora_sparsity: float = 0.1  # Top-10% weights kept
    sim_threshold: float = 0.7  # Cosine similarity threshold for mask sharing
    force_sharing_test: bool = False  # Force sharing/composition for testing

class Runner:
    def __init__(self, all_args: Args, train_xlsx=None, test_xlsx=None, Q_emb=None, Q_mask=None, barrier=None):
        self.args = all_args
        self.train_xlsx = train_xlsx  # Store log directory for Excel output
        self.test_xlsx = test_xlsx
        self.Q_emb = Q_emb
        self.Q_mask = Q_mask
        self.barrier = barrier
        self.all_envs = all_args.all_envs.split(",") if all_args.all_envs else [all_args.env_id]
        self.task_idx = self.all_envs.index(all_args.env_id) if all_args.env_id in self.all_envs else 0

        assert self.args.alg_name in ["ppo", "grpo"]
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        wandb.init(
            config=all_args.__dict__,
            project="RLVLA-MOSAIC",
            name=self.args.name,
            mode="online" if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        # Policy with frozen backbone and LoRA
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)
        # Freeze backbone
        # Freeze only backbone parameters, keep LoRA and value head trainable
        for name, param in self.policy.vla.named_parameters():
            if "lora" in name or "value_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.lora_params = [p for p in self.policy.vla.parameters() if p.requires_grad]  # LoRA only
        self.sparse_lora = None  # Cache sparse LoRA
        self.composed_params = None  # Cache composed LoRA
        self.beta_weights = [1.0] + [0.0] * (len(self.all_envs) - 1)  # Initialize: own mask weight=1, others=0

        self.alg = OpenVLAPPO(all_args, self.policy)
        unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        self.env = SimlerWrapper(self.args, unnorm_state)
        
        # PPO buffer (on-policy)
        self.buffer = FifoReplayBuffer(
            obs_dim=(480, 640, 3),
            act_dim=7,
            capacity=100000,
            episode_len=self.args.episode_len,
            num_envs=self.args.num_envs
        )
        # FIFO buffer (off-policy/replay/embedding)
        from simpler_env.utils.replay_buffer import FifoReplayBuffer as FifoReplayBufferOffPolicy
        self.fifo_buffer = FifoReplayBufferOffPolicy(
            capacity=10000,
            obs_shape=(480, 640, 3),
            action_shape=(7,),
            dtype_obs=np.uint8,
            dtype_action=np.float32
        )
        self.last_obs = None  # For storing the last observation for next_obs
        self.last_instruction = None
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Memory after env init: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        # MOSAIC: Initialize task embedding and performance
        self.task_embedding = None
        self.performance = 0.0  # Mean reward
        self.received_masks = {}  # Dict of {agent_id: LoRA_params}
        # Remove shared lists; use multiprocessing queues for embeddings and masks

    def compute_task_embedding(self, buffer):
        """Compute Wasserstein Task Embedding from SAR transitions sampled from the given buffer."""
        num_samples = 100
        if len(buffer) < num_samples:
            raise ValueError("Not enough samples in buffer to compute task embedding.")
        batch = buffer.sample(num_samples)
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]

        # Flatten and normalize (assumes images flattened)
        obs_flat = obs.reshape(num_samples, -1) / 255.0
        actions_flat = actions.reshape(num_samples, -1)
        rewards_flat = rewards.reshape(num_samples)

        if actions_flat.size == 0:
            raise ValueError("Actions array is empty in compute_task_embedding. Cannot normalize.")
        actions_flat = actions_flat / (np.max(np.abs(actions_flat)) + 1e-6)
        if rewards_flat.size == 0:
            raise ValueError("Rewards array is empty in compute_task_embedding. Cannot normalize.")
        rewards_flat = rewards_flat / (np.max(np.abs(rewards_flat)) + 1e-6)

        sar = np.concatenate([obs_flat, actions_flat, rewards_flat[:, None]], axis=1)
        mu_tau = sar / (np.sum(sar, axis=1, keepdims=True) + 1e-6)  # Normalize to distribution
        mu_0 = np.ones_like(mu_tau) / mu_tau.shape[1]
        M = ot.dist(mu_tau, mu_0, metric='euclidean')
        v_tau = ot.emd2([], [], M)  # Embedding as Wasserstein vector
        return torch.tensor([v_tau], dtype=torch.float32, device=self.device)

    def make_sparse_lora(self):
        """Apply top-k sparsity to LoRA parameters."""
        if not self.sparse_lora:
            self.sparse_lora = []
            for param in self.lora_params:
                param_flat = param.view(-1)
                k = int(param_flat.numel() * self.args.lora_sparsity)
                _, indices = torch.topk(param_flat.abs(), k=k, largest=True)
                mask = torch.zeros_like(param_flat)
                mask[indices] = 1.0
                sparse_param = param * mask.view_as(param)
                self.sparse_lora.append(sparse_param)
        return self.sparse_lora

    def compose_policy(self):
        """Compose policy with own and peer LoRA masks."""
        own_lora = self.make_sparse_lora()
        composed = []
        for i, param in enumerate(own_lora):
            weighted_sum = param * self.beta_weights[self.task_idx]
            for agent_id, peer_lora in self.received_masks.items():
                peer_idx = agent_id  # Assuming agent_id corresponds to task_idx sequentially
                weighted_sum += peer_lora[i] * self.beta_weights[peer_idx]
            composed.append(weighted_sum)
        self.composed_params = composed
        # Update policy parameters (apply to LoRA layers)
        lora_idx = 0
        for param in self.policy.vla.parameters():
            if param.requires_grad:
                param.data.copy_(self.composed_params[lora_idx])
                lora_idx += 1

    def share_and_receive(self, episode):
        logging.info(f"[share_and_receive] Called at episode {episode}")
        try:
            if episode % self.args.comm_interval != 0:
                logging.info("[share_and_receive] Not a communication interval; returning early")
                return
            logging.info(f"[share_and_receive] Communication interval hit (comm_interval={self.args.comm_interval})")
            # Only compute embedding if FIFO buffer has data (standardize to step for consistency)
            if len(self.fifo_buffer) == 0:
                logging.warning(f"[share_and_receive] Skipping: FIFO buffer has no data yet. size={len(self.fifo_buffer)}")
                if self.barrier:
                    logging.debug("[share_and_receive] Waiting on barriers during skip")
                    try:
                        self.barrier.wait(timeout=60)  # For TEQ phase
                    except Exception as e:
                        logging.error("[share_and_receive] Barrier wait (TEQ phase) failed: %s", e, exc_info=True)
                        try:
                            self.barrier.abort()
                        except Exception as be:
                            logging.error("[share_and_receive] Failed to abort barrier (TEQ phase): %s", be, exc_info=True)
                        raise
                    try:
                        self.barrier.wait(timeout=60)  # For mask phase
                    except Exception as e:
                        logging.error("[share_and_receive] Barrier wait (mask phase) failed: %s", e, exc_info=True)
                        try:
                            self.barrier.abort()
                        except Exception as be:
                            logging.error("[share_and_receive] Failed to abort barrier (mask phase): %s", be, exc_info=True)
                        raise
                    try:
                        self.barrier.wait(timeout=60)  # Extra for clear phase
                    except Exception as e:
                        logging.error("[share_and_receive] Barrier wait (clear phase) failed: %s", e, exc_info=True)
                        try:
                            self.barrier.abort()
                        except Exception as be:
                            logging.error("[share_and_receive] Failed to abort barrier (clear phase): %s", be, exc_info=True)
                        raise
                return
            logging.info("[share_and_receive] FIFO buffer has data; proceeding to compute embedding and performance")
            self.task_embedding = self.compute_task_embedding(self.fifo_buffer)
            logging.debug(f"[share_and_receive] Task embedding computed: shape={self.task_embedding.shape}, first few values={self.task_embedding[:5]}")  # Log sample for inspection
            # Compute performance as mean reward from the most recent transitions in FIFO buffer
            rewards = []
            # Try to get the most recent N rewards (N=100 for consistency with embedding)
            N = 100
            if len(self.fifo_buffer) >= N:
                # Assume buffer has a method to get the last N rewards, else sample N
                try:
                    # If buffer supports direct access to last N
                    rewards = [self.fifo_buffer.rewards[(self.fifo_buffer.ptr - i - 1) % self.fifo_buffer.capacity][0] for i in range(N)]
                except Exception:
                    # Fallback: sample N
                    batch = self.fifo_buffer.sample(N)
                    rewards = batch["rewards"].flatten()
            else:
                # Use all available rewards
                try:
                    rewards = [self.fifo_buffer.rewards[i][0] for i in range(len(self.fifo_buffer))]
                except Exception:
                    batch = self.fifo_buffer.sample(len(self.fifo_buffer))
                    rewards = batch["rewards"].flatten()
            self.performance = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
            logging.info(f"[share_and_receive] Computed performance: {self.performance} (mean reward from FIFO buffer, N={len(rewards)})")
        except Exception as e:
            logging.error(f"[share_and_receive] Exception before barrier: {e}", exc_info=True)
            if self.barrier:
                try:
                    self.barrier.abort()
                except Exception as be:
                    logging.error(f"[share_and_receive] Failed to abort barrier: {be}", exc_info=True)
            raise

        # --- FORCE SHARING/COMPOSITION FOR TESTING ---
        if getattr(self.args, "force_sharing_test", False):
            
            # Add a fake peer TEQ with high similarity and lower performance
            peer_id = 1 if self.args.agent_id == 0 else 0
            noise = torch.randn_like(self.task_embedding) * 0.01
            peer_embedding = (self.task_embedding + noise).cpu().numpy()
            peer_perf = max(self.performance - 0.5, 0.1)
            if self.Q_emb is not None:
                self.Q_emb.put((peer_id, peer_embedding, peer_perf, episode))
            # Add a second peer with higher performance to test receiving
            peer_id2 = 2 if self.args.agent_id == 0 else 0
            peer_embedding2 = (self.task_embedding + noise * 2).cpu().numpy()
            peer_perf2 = self.performance + 0.5
            if self.Q_emb is not None:
                self.Q_emb.put((peer_id2, peer_embedding2, peer_perf2, episode))
            logging.info(f"[force_sharing_test] Injected fake peer TEQs for testing sharing/composition.")

        # Share TEQ via shared list (include episode for filtering)
        teq = (self.args.agent_id, self.task_embedding.cpu().numpy(), self.performance, episode)
        if self.Q_emb is not None:
            self.Q_emb.put(teq)
        logging.info(f"[share_and_receive] Put TEQ to Q_emb: agent_id={self.args.agent_id}, embedding_shape={teq[1].shape}, performance={teq[2]}, episode={teq[3]}")

        if self.barrier:
            logging.debug("[share_and_receive] Waiting on TEQ barrier")
            self.barrier.wait()

        # Read all current TEQs (filter by episode to avoid stale data)
        # Drain Q_emb and collect TEQs for this episode
        current_teqs = []
        if self.Q_emb is not None:
            try:
                while True:
                    t = self.Q_emb.get_nowait()
                    if t[3] == episode:
                        current_teqs.append(t)
            except Exception:
                pass
        logging.info(f"[share_and_receive] Read {len(current_teqs)} TEQs from Q_emb for episode {episode}")
        received_teqs = []
        teq_count = 0
        peer_details = []
        for peer_id, peer_emb_np, peer_perf, _ in current_teqs:
            if peer_id == self.args.agent_id:
                logging.debug(f"[share_and_receive] Skipping own TEQ: peer_id={peer_id}")
                continue
            peer_emb = torch.tensor(peer_emb_np, device=self.device)
            cos_sim = torch.cosine_similarity(self.task_embedding, peer_emb, dim=0)
            is_similar = cos_sim.item() > self.args.sim_threshold
            is_better = peer_perf > self.performance
            peer_details.append({
                "peer_id": peer_id,
                "peer_perf": peer_perf,
                "cos_sim": cos_sim.item(),
                "is_similar": is_similar,
                "is_better": is_better
            })
            logging.info(f"[share_and_receive] Processed peer {peer_id}: cosine_sim={cos_sim.item():.4f}, peer_perf={peer_perf}")
            if is_similar:
                received_teqs.append((peer_id, peer_emb, peer_perf))
                logging.info(f"[share_and_receive] Added similar peer {peer_id} (sim > {self.args.sim_threshold})")
            teq_count += 1
        # Print peer summary table
        if peer_details:
            logging.info("[share_and_receive] Peer summary (id | perf | cos_sim | is_similar | is_better):")
            for d in peer_details:
                logging.info(f"  id={d['peer_id']} | perf={d['peer_perf']:.4f} | cos_sim={d['cos_sim']:.4f} | similar={d['is_similar']} | better={d['is_better']}")
        else:
            logging.info("[share_and_receive] No peers found for summary.")
        logging.info(f"[share_and_receive] Processed {teq_count} peer TEQs, found {len(received_teqs)} similar peers")

        # Send masks if better (append to shared list, include episode)
        serialized_lora = None
        sent_masks = 0
        for peer_id, _, peer_perf in received_teqs:
            if self.performance > peer_perf:
                
                logging.info(f"[share_and_receive] Better than peer {peer_id} (my_perf={self.performance} > peer_perf={peer_perf}); preparing to send mask")
                
                if serialized_lora is None:
                    sparse_lora = self.make_sparse_lora()
                    logging.debug(f"[share_and_receive] Created sparse LoRA: num_params={len(sparse_lora)}, example_shape={sparse_lora[0].shape if sparse_lora else 'empty'}")
                    serialized_lora = [p.detach().cpu().to(torch.float32).numpy() for p in sparse_lora]
                    logging.debug(f"[share_and_receive] Serialized LoRA: num_arrays={len(serialized_lora)}")
                
                mask_data = (self.args.agent_id, serialized_lora, episode)
                if self.Q_mask is not None:
                    self.Q_mask.put(mask_data)
                
                sent_masks += 1
                logging.info(f"[share_and_receive] Appended mask to mask_list for peer {peer_id}")

        logging.info(f"[share_and_receive] Sent {sent_masks} masks in total")

        if self.barrier:
            logging.debug("[share_and_receive] Waiting on mask barrier")
            try:
                self.barrier.wait(timeout=60)
            except Exception as e:
                logging.error(f"[share_and_receive] Barrier wait failed: {e}", exc_info=True)
                # Optionally, abort the barrier to unblock others
                try:
                    self.barrier.abort()
                except Exception as be:
                    logging.error(f"[share_and_receive] Failed to abort barrier: {be}", exc_info=True)
                raise

        # Read all current masks
        # Drain Q_mask and collect masks for this episode
        current_masks = []
        if self.Q_mask is not None:
            try:
                while True:
                    m = self.Q_mask.get_nowait()
                    if m[2] == episode:
                        current_masks.append(m)
            except Exception:
                pass
        logging.info(f"[share_and_receive] Read {len(current_masks)} masks from Q_mask for episode {episode}")
        
        better_peer_ids = [p_id for p_id, _, p_perf in received_teqs if p_perf > self.performance]
        logging.info(f"[share_and_receive] Identified better peers: {better_peer_ids}")
        
        self.received_masks = {}
        mask_count = len(current_masks)
        
        for sender_id, peer_lora_ser, _ in current_masks:
            if sender_id in better_peer_ids:
                logging.info(f"[share_and_receive] Accepting mask from better peer {sender_id}")
                peer_lora = [torch.tensor(p, device=self.device) for p in peer_lora_ser]
                logging.debug(f"[share_and_receive] Deserialized peer LoRA: num_tensors={len(peer_lora)}, example_shape={peer_lora[0].shape if peer_lora else 'empty'}")
                self.received_masks[sender_id] = peer_lora
            else:
                logging.debug(f"[share_and_receive] Ignoring mask from non-better peer {sender_id}")
        logging.info(f"[share_and_receive] Total masks in list: {mask_count}, received and stored {len(self.received_masks)}")

        # Update beta weights
        peer_id_to_perf = {p_id: p_perf for p_id, _, p_perf in received_teqs}
        logging.debug(f"[share_and_receive] Peer performance map: {peer_id_to_perf}")
        total_perf = self.performance
        for peer_id in self.received_masks:
            peer_perf = peer_id_to_perf.get(peer_id, 0)
            total_perf += peer_perf
            logging.debug(f"[share_and_receive] Adding perf from peer {peer_id}: {peer_perf}")
        logging.info(f"[share_and_receive] Calculated total_perf={total_perf}")
        if total_perf > 0:
            self.beta_weights[self.task_idx] = self.performance / (total_perf + 1e-6)
            logging.info(f"[share_and_receive] Updated self beta_weight (idx {self.task_idx}): {self.beta_weights[self.task_idx]}")
            for peer_id in self.received_masks:
                peer_idx = peer_id  # Assuming sequential assignment
                self.beta_weights[peer_idx] = peer_id_to_perf[peer_id] / (total_perf + 1e-6)
                logging.info(f"[share_and_receive] Updated beta_weight for peer {peer_id} (idx {peer_idx}): {self.beta_weights[peer_idx]}")
        else:
            logging.warning("[share_and_receive] Total performance <= 0; beta_weights unchanged")
        logging.info(f"[share_and_receive] Final beta_weights: {self.beta_weights}")

        logging.info("[share_and_receive] Starting policy composition")
        self.compose_policy()
        logging.info("[share_and_receive] Composed policy with new beta_weights and received masks")

        # Clear lists safely (extra barrier, then leader clears)
        if self.barrier:
            logging.debug("[share_and_receive] Waiting on clear barrier")
            try:
                self.barrier.wait(timeout=60)
            except Exception as e:
                logging.error("[share_and_receive] Barrier wait (final clear phase) failed: %s", e, exc_info=True)
                try:
                    self.barrier.abort()
                except Exception as be:
                    logging.error("[share_and_receive] Failed to abort barrier (final clear phase): %s", be, exc_info=True)
                raise
            if self.args.agent_id == 0:
                # No need to clear queues; they are drained each round
                logging.info("[share_and_receive] Leader: queues are used for communication, no explicit clear needed")
            else:
                logging.debug("[share_and_receive] Non-leader waiting for clear")

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]
        values, actions, logprobs = [], [], []
        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
        return (
            torch.cat(values, dim=0).to(device=self.device),
            torch.cat(actions, dim=0).to(device=self.device),
            torch.cat(logprobs, dim=0).to(device=self.device)
        )

    def collect(self):
        self.policy.prep_rollout()
        # Get the most recent observation (ptr points to next insert, so last is ptr-1)
        # Use the last step for PPO buffer
        idx = (self.buffer.step) % self.buffer.episode_len
        obs_image = self.buffer.observations[idx]
        obs_image = torch.tensor(obs_image).to(self.device)
        # Add batch dimension and select the corresponding instruction
        obs_image = obs_image.unsqueeze(0)  # shape: [1, 480, 640, 3]
        # Use the instruction for the first environment (or all if needed)
        instruction = [self.buffer.instruction[0]]
        obs = dict(image=obs_image, task_description=instruction)
        value, action, logprob = self._get_action(obs)
        return value, action, logprob

    def insert(self, data):
        # data: (obs_img, actions, logprob, value_preds, rewards, done)
        obs_img, actions, logprob, value_preds, rewards, done = data
        obs_img_np = obs_img.cpu().numpy()
        actions_np = actions.cpu().numpy()
        logprob_np = logprob.cpu().numpy()
        value_preds_np = value_preds.cpu().to(torch.float32).numpy()
        rewards_np = rewards.cpu().numpy()
        done_np = done.cpu().numpy()
        # Insert the full batch for PPO
        self.buffer.insert_ppo(
            obs=obs_img_np,
            actions=actions_np,
            action_log_probs=logprob_np,
            value_preds=value_preds_np,
            rewards=rewards_np,
            masks=1.0 - done_np,  # masks: 1.0 if not done, 0.0 if done
            instruction=self.last_instruction if self.last_instruction is not None else None
        )
        # Also insert each transition into the FIFO buffer for replay/embedding
        # Use obs_img as current obs, actions, rewards, next_obs (from next step), done, instruction
        # Since next_obs is not available here, use obs_img_np (current) for both obs and next_obs for now
        # Ideally, pass next_obs from the environment step
        batch_size = actions_np.shape[0]
        for i in range(batch_size):
            self.fifo_buffer.insert(
                obs=obs_img_np[i],
                action=actions_np[i],
                reward=rewards_np[i],
                next_obs=obs_img_np[i],  # If you have next_obs, use it here
                done=done_np[i],
                info=self.last_instruction[i] if self.last_instruction is not None else ""
            )
        self.last_obs = obs_img_np
        self.last_instruction = self.last_instruction if self.last_instruction is not None else [""] * obs_img_np.shape[0]
        print(f"[Runner] PPO Buffer step: {self.buffer.step} | FIFO Buffer size: {len(self.fifo_buffer)}")

    def compute_endup(self):
        self.policy.prep_rollout()
        # Use the last step for PPO buffer
        idx = (self.buffer.step) % self.buffer.episode_len
        obs_image = torch.tensor(self.buffer.observations[idx]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        # No endup method for FifoReplayBuffer; skip this step
        next_value = next_value.to(torch.float32).cpu().numpy()
        # self.buffer.endup(next_value)  # Not applicable for FifoReplayBuffer

    def train(self):
        self.policy.prep_training()
        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")
        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        # No masks in FifoReplayBuffer; skip mask_mean
        return info

    @torch.no_grad()
    def eval(self, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])
        obs_img, instruction, info = self.env.reset(obj_set=obj_set)
        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)
            obs_img, reward, done, env_info, instruction = self.env.step(action)
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")
        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],
            "instruction": "",
            "action": [],
            "info": [],
        } for idx in range(self.args.num_envs)]
        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]
        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)
            obs_img_new, reward, done, env_info, instruction = self.env.step(action)
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v
            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)
            obs_img = obs_img_new
            # instruction is updated from env.step, so it will be used in the next loop iteration
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1
            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )
            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}",
                            fps=10, verbose=False)
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }
        save_stats = {
            "env_name": self.args.env_id,
            "ep_len": self.args.episode_len,
            "epoch": epoch,
            "stats": {k: v.item() for k, v in env_stats.items()},
            "instruction": {idx: ins for idx, ins in enumerate(instruction)},
            "last_info": last_info,
        }
        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))
        return env_stats

    def run(self):
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        
        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            # Validation: Ensure obs_img and instruction match in batch size
            if hasattr(obs_img, "shape") and isinstance(instruction, list):
                assert obs_img.shape[0] == len(instruction), (
                    f"[Runner] Mismatch after reset: obs_img.shape[0]={obs_img.shape[0]}, len(instruction)={len(instruction)}"
                )
            # FIFO buffer does not need warmup; instead, set last_obs and last_instruction for transition tracking
            self.last_obs = obs_img.cpu().numpy()
            self.last_instruction = instruction
            for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                # Generate actions for all environments in the batch
                obs = dict(image=torch.tensor(obs_img).to(self.device), task_description=self.last_instruction)
                value, action, logprob = self._get_action(obs)
                obs_img_new, reward, done, env_info, instruction = self.env.step(action)
                # Validation: Ensure obs_img_new and instruction match in batch size
                if hasattr(obs_img_new, "shape") and isinstance(instruction, list):
                    assert obs_img_new.shape[0] == len(instruction), (
                        f"[Runner] Mismatch after step: obs_img_new.shape[0]={obs_img_new.shape[0]}, len(instruction)={len(instruction)}"
                    )
                data = (obs_img_new, action, logprob, value, reward, done)
                self.insert(data)
                # Update last_obs and last_instruction for next step
                self.last_obs = obs_img_new.cpu().numpy()
                # Instruction is not updated by env.step(), so keep previous
                obs_img = obs_img_new.cpu().numpy()
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs
            
            print(pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))
            
            self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            gc.collect()
            torch.cuda.empty_cache()
            
            # MOSAIC: Share and receive masks
            self.share_and_receive(episode)
            infos = self.train()
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)
            wandb.log(infos, step=steps)
            elapsed_time = time.time() - ep_time
            print(f"{self.args.name}: ep {episode:0>4d} | steps {steps} | e {elapsed_time:.2f}s")
            print(pprint.pformat({k: round(v, 4) for k, v in infos.items()}))
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                def aggregate_eval(runs):
                    keys = runs[0].keys()
                    mean = {k: np.mean([d[k] for d in runs]) for k in keys}
                    std = {k: np.std([d[k] for d in runs]) for k in keys}
                    return mean, std
                train_eval_runs = [self.eval(obj_set="train") for _ in range(self.args.num_eval_runs)]
                train_mean, train_std = aggregate_eval(train_eval_runs)
                sval_stats = {f"eval/{k}": v for k, v in train_mean.items()}
                sval_stats.update({f"eval/{k}_std": train_std[k] for k in train_mean})
                wandb.log(sval_stats, step=steps)
                print("Train eval mean:", pprint.pformat({k: round(v, 4) for k, v in train_mean.items()}))
                print("Train eval std:", pprint.pformat({k: round(v, 4) for k, v in train_std.items()}))

                # --- Append to train Excel ---
                if self.train_xlsx is not None:
                    
                    
                    # Only append steps and mean_success
                    mean_success = None
                    for k, v in train_mean.items():
                        if "success" in k:
                            mean_success = v
                            break
                    train_row = {"steps": steps, "mean_success": mean_success}
                    try:
                        df = pd.read_excel(self.train_xlsx)
                        df = pd.concat([df, pd.DataFrame([train_row])], ignore_index=True)
                    except Exception:
                        df = pd.DataFrame([train_row])
                        
                    print(df)
                    train_xlsx_path = Path(self.train_xlsx)
                    train_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_excel(self.train_xlsx, index=False)
                    logging.info(f"Appended to train.xlsx: steps={steps}, mean_success={mean_success}")

                # For "test" set
                test_eval_runs = [self.eval(obj_set="test") for _ in range(self.args.num_eval_runs)]
                test_mean, test_std = aggregate_eval(test_eval_runs)
                sval_stats = {f"eval/{k}_ood": v for k, v in test_mean.items()}
                sval_stats.update({f"eval/{k}_ood_std": test_std[k] for k in test_mean})
                wandb.log(sval_stats, step=steps)
                print("Test eval mean:", pprint.pformat({k: round(v, 4) for k, v in test_mean.items()}))
                print("Test eval std:", pprint.pformat({k: round(v, 4) for k, v in test_std.items()}))

                # --- Append to test Excel ---
                if self.test_xlsx is not None:
                    
                    # Only append steps and mean_success
                    mean_success = None
                    for k, v in test_mean.items():
                        if "success" in k:
                            mean_success = v
                            break
                    test_row = {"steps": steps, "mean_success": mean_success}
                    try:
                        df = pd.read_excel(self.test_xlsx)
                        df = pd.concat([df, pd.DataFrame([test_row])], ignore_index=True)
                    except Exception:
                        df = pd.DataFrame([test_row])
                    test_xlsx_path = Path(self.test_xlsx)
                    test_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_excel(self.test_xlsx, index=False)
                    logging.info(f"Appended to test.xlsx: steps={steps}, mean_success={mean_success}")

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                #self.policy.save(save_path)

                self.render(epoch=episode, obj_set="train")
                self.render(epoch=episode, obj_set="test")

def main():
    args = tyro.cli(Args)
    # --- Logging and logdir setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / timestamp / args.env_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.txt"

    # Create empty train.xlsx and test.xlsx at the beginning
    train_xlsx = log_dir / "train.xlsx"
    test_xlsx = log_dir / "test.xlsx"
    pd.DataFrame().to_excel(train_xlsx, index=False)
    pd.DataFrame().to_excel(test_xlsx, index=False)

    # Redirect stdout and stderr to log.txt
    log_fh = open(log_file, "a")
    sys.stdout = log_fh
    sys.stderr = log_fh

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(log_fh)
        ]
    )
    logging.info("Logging started. Log file: %s", log_file)

    
    

    Q_emb = mp.Queue() if args.comm_interval > 0 else None
    Q_mask = mp.Queue() if args.comm_interval > 0 else None
    runner = Runner(args,train_xlsx, test_xlsx, Q_emb, Q_mask)
    
    if args.only_render:
        ll = [
            "PutOnPlateInScene25VisionImage-v1",
            "PutOnPlateInScene25VisionTexture03-v1",
            "PutOnPlateInScene25VisionTexture05-v1",
            "PutOnPlateInScene25VisionWhole03-v1",
            "PutOnPlateInScene25VisionWhole05-v1",
            "PutOnPlateInScene25Instruct-v1",
            "PutOnPlateInScene25Plate-v1",
            "PutOnPlateInScene25Position-v1",
            "PutOnPlateInScene25EEPose-v1",
            "PutOnPlateInScene25PositionChange-v1",
            "PutOnPlateInScene25PositionChangeTo-v1"
        ]
        if args.env_id not in ll:
            runner.render(epoch=0, obj_set="train")
        runner.render(epoch=0, obj_set="test")
    else:
        runner.run()
    # Close log file at end
    log_fh.close()

if __name__ == "__main__":
    main()