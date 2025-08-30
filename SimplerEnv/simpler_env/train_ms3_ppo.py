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
import logging #@/SimplerEnv/simpler_env/train_ms3_ppo.py add one more buffer with fixed capacity called buffer_fifo next to existing buffer. use this buffer to compute embeddings which is used for calculate task similarity and sharing
import ot   #For Wasserstein embeddings
import pickle
import multiprocessing as mp

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

signal.signal(signal.SIGINT, signal.SIG_DFL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
from dotenv import load_dotenv
import wandb

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("WANDB_API_KEY")

# Login to Weights & Biases
wandb.login(key=api_key)

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
    steps_max: int = 2000000
    steps_vh: int = 0
    interval_eval: int = 3
    interval_save: int = 40
    buffer_inferbatch: int = 4  #for rollout just pass chunks of env data to save memory
    buffer_minibatch: int = 2   #for training just pass chunks of stored buffer samples to save memory  
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
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False
    num_eval_runs: int = 1
    # MOSAIC-specific args
    force_sharing_test: bool = False
    comm_interval: int = 2
    agent_id: int = 0
    all_envs: str = ""
    lora_sparsity: float = 1.0  # Top-10% weights kept
    sim_threshold: float = 0.5  # Cosine similarity threshold for mask sharing

class Runner:
    def __init__(self, all_args: Args, train_xlsx=None, test_xlsx=None, shared_teqs=None, shared_masks=None, barrier=None, manager=None):
        self.args = all_args
        self.train_xlsx = train_xlsx  # Store log directory for Excel output
        self.test_xlsx = test_xlsx
        # self.Q_emb = Q_emb
        # self.Q_mask = Q_mask
        self.shared_teqs = shared_teqs
        self.shared_masks = shared_masks
        self.manager = manager
        
        self.barrier = barrier
        self.all_envs = all_args.all_envs.split(",") if all_args.all_envs else [all_args.env_id]
        self.task_idx = self.all_envs.index(all_args.env_id) if all_args.env_id in self.all_envs else 0

        assert self.args.alg_name in ["ppo", "grpo"]
        
        # Set random seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # Initialize wandb
        wandb.init(
            config=all_args.__dict__,
            project="RLVLA-MOSAIC",
            name=self.args.env_id + "_" + str(self.args.seed),
            mode="online" if self.args.wandb else "offline",
            reinit=True,
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
        
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
        )
        
        # Add FIFO buffer for embeddings/task similarity
        from simpler_env.utils.replay_buffer import FifoReplayBuffer
        self.buffer_fifo = FifoReplayBuffer(
            capacity=1000,  # Fixed capacity for embedding buffer
            obs_shape=(480, 640, 3),
            action_shape=(7,),
            dtype_obs=np.uint8,
            dtype_action=np.int32
        )

        # MOSAIC: Initialize task embedding and performance
        self.task_embedding = None
        self.performance = 0.0  # Mean reward
        self.received_masks = {}  # Dict of {agent_id: LoRA_params}

    def compute_task_embedding_(self):
        """Compute Wasserstein Task Embedding from SAR trajectories using buffer_fifo."""
        # Sample SAR tuples from buffer_fifo
        num_samples = min(100, len(self.buffer_fifo))
        if num_samples == 0:
            raise ValueError("buffer_fifo is empty in compute_task_embedding. Cannot compute embedding.")
        # Randomly sample indices
        idxs = np.random.choice(len(self.buffer_fifo), num_samples, replace=False)#replace false means no duplicates
        obs = self.buffer_fifo.obs[idxs]
        actions = self.buffer_fifo.actions[idxs]
        rewards = self.buffer_fifo.rewards[idxs]
        print("rewards", rewards)
        # Flatten and normalize (assumes images flattened)
        states = np.array([o.flatten() for o in obs]) / 255.0
        
        # Handle empty actions array to avoid ValueError
        if actions.size == 0:
            raise ValueError("Actions array is empty in compute_task_embedding. Cannot normalize.")
        actions_flat = actions / (np.max(np.abs(actions)) + 1e-6)
        if rewards.size == 0:
            raise ValueError("Rewards array is empty in compute_task_embedding. Cannot normalize.")
        rewards_flat = rewards / (np.max(np.abs(rewards)) + 1e-6)
        
        sar = np.concatenate([states, actions_flat, rewards_flat[:, None]], axis=1)
        mu_tau = sar / (np.sum(sar, axis=1, keepdims=True) + 1e-6)  # Normalize to distribution
        # Reference distribution (simplified: uniform)
        mu_0 = np.ones_like(mu_tau) / mu_tau.shape[1]
        # Wasserstein distance
        M = ot.dist(mu_tau, mu_0, metric='euclidean')
        v_tau = ot.emd2([], [], M)  # Embedding as Wasserstein vector
        return torch.tensor(v_tau, dtype=torch.float32, device=self.device)
    
 

    def compute_task_embedding(self, num_samples=128, M_ref=None, use_sinkhorn=True, sinkhorn_reg=1e-2):
        """
        Compute WTE embedding (barycenter projection) using samples from buffer_fifo.
        Returns: torch.Tensor of shape (M_ref * d,) (flattened) or (M_ref, d) if reshape=False
        Requirements: self.wte_reference must exist (M_ref x d numpy array), created once at init.
        """
        # 1) sample
        num_available = len(self.buffer_fifo)
        num_samples = min(num_samples, num_available)
        if num_samples == 0:
            raise ValueError("buffer_fifo is empty in compute_task_embedding. Cannot compute embedding.")

        idxs = np.random.choice(num_available, num_samples, replace=False)
        obs = self.buffer_fifo.obs[idxs]
        actions = self.buffer_fifo.actions[idxs]
        rewards = self.buffer_fifo.rewards[idxs]

        # 2) build X (N x d)
        # states: flatten images and scale to [0,1]
        states = np.array([o.flatten() for o in obs], dtype=np.float64) / 255.0  # (N, ds)
        # actions: normalize by max abs across batch (per your code)
        if actions.size == 0:
            raise ValueError("Actions array is empty in compute_task_embedding. Cannot normalize.")
        
        actions = actions.astype(np.float64)
        max_act = np.max(np.abs(actions)) + 1e-9
        actions_flat = actions / max_act
        if actions_flat.ndim == 1:
            actions_flat = actions_flat[:, None]  # ensure shape (N, da)
        # rewards: normalize similarly
        if rewards.size == 0:
            raise ValueError("Rewards array is empty in compute_task_embedding. Cannot normalize.")
        rewards = rewards.astype(np.float64)
        max_r = np.max(np.abs(rewards)) + 1e-9
        rewards_flat = (rewards / max_r)[:, None]

        # Concatenate features -> X (N x d)
        X = np.concatenate([states, actions_flat, rewards_flat], axis=1)  # dtype float64
        N, d = X.shape

        # 3) reference points
        if M_ref is None:
            if not hasattr(self, 'wte_reference'):
                # initialize a reference set of M anchors in same feature range
                M_ref = 50
                # sample uniform in [-1,1]^d and then scale appropriately (or sample from data mean)
                self.wte_reference = np.random.uniform(-1.0, 1.0, size=(M_ref, d)).astype(np.float64)
            else:
                M_ref = self.wte_reference.shape[0]
        else:
            if not hasattr(self, 'wte_reference'):
                self.wte_reference = np.random.uniform(-1.0, 1.0, size=(M_ref, d)).astype(np.float64)

        X0 = self.wte_reference  # shape (M_ref, d)

        # 4) weights
        a = np.ones(N, dtype=np.float64) / N  # source weights
        b = np.ones(M_ref, dtype=np.float64) / M_ref  # target (reference) weights

        # 5) cost matrix (squared Euclidean for 2-Wasserstein)
        C = ot.dist(X, X0, metric='euclidean') ** 2  # shape (N, M_ref)

        # 6) transport plan: use Sinkhorn (faster and regularized) or exact EMD
        if use_sinkhorn:
            # sinkhorn returns transport matrix (N x M_ref)
            gamma = ot.sinkhorn(a, b, C, reg=sinkhorn_reg)  # shape (N, M_ref)
        else:
            gamma = ot.emd(a, b, C)  # exact; might be slower

        # 7) barycenter projection -> embedding matrix V (M_ref x d)
        # gamma is N x M_ref, we need gamma.T @ X -> (M_ref x d)
        V = gamma.T.dot(X)  # shape (M_ref, d)

        # optionally flatten to vector
        v_flat = V.ravel()  # shape (M_ref * d,)
        return torch.tensor(v_flat, dtype=torch.float32, device=self.device)


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
                peer_idx = self.all_envs.index(ENVIRONMENTS[agent_id]) if agent_id < len(ENVIRONMENTS) else 0
                weighted_sum += peer_lora[i] * self.beta_weights[peer_idx]
            composed.append(weighted_sum)
        self.composed_params = composed
        # Update policy parameters (apply to LoRA layers)
        lora_idx = 0
        for param in self.policy.vla.parameters():
            if param.requires_grad:
                param.data.copy_(self.composed_params[lora_idx])
                lora_idx += 1
                
                
    # --- Utilities for safe append to manager dict of lists ---
    def _ensure_episode_list(self, shared_dict, episode):
        """Ensure shared_dict[episode] exists and is a manager.list()."""
        if episode not in shared_dict:
            # race: multiple agents may try to set; last write wins but all will append to the same list proxy object type
            shared_dict[episode] = self.manager.list()
        return shared_dict[episode]

    def share_and_receive(self, episode, current_success):
        logging.info(f"[share_and_receive] Called at episode {episode}")
        
        if len(self.buffer_fifo) > 0:
            logging.info("[share_and_receive] FIFO buffer has data; proceeding to compute embedding and performance")
            self.task_embedding = self.compute_task_embedding()
            logging.info(f"[share_and_receive] Task embedding computed: shape={self.task_embedding.shape}, first few values={self.task_embedding}")  # Log sample for inspection
            
        try:
            if episode % self.args.comm_interval != 0:
                logging.info("[share_and_receive] Not a communication interval; returning early")
                return
            logging.info(f"[share_and_receive] Communication interval hit (comm_interval={self.args.comm_interval})")
            # Only compute embedding if FIFO buffer has data (standardize to step for consistency)
            if len(self.buffer_fifo) == 0:
                logging.warning(f"[share_and_receive] Skipping: FIFO buffer has no data yet. size={len(self.buffer_fifo)}")
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
            
            
            
            
            #=======================================================
            # Compute performance as mean reward from the most recent transitions in FIFO buffer
            rewards = []
            # Try to get the most recent N rewards (N=100 for consistency with embedding)
            N = 100
            if len(self.buffer_fifo) >= N:
                # Assume buffer has a method to get the last N rewards, else sample N
                try:
                    # If buffer supports direct access to last N
                    rewards = [self.buffer_fifo.rewards[(self.buffer_fifo.ptr - i - 1) % self.buffer_fifo.capacity][0] for i in range(N)]
                except Exception:
                    # Fallback: sample N
                    batch = self.buffer_fifo.sample(N)
                    rewards = batch["rewards"].flatten()      
            else:
                # Use all available rewards
                try:
                    rewards = [self.buffer_fifo.rewards[i][0] for i in range(len(self.buffer_fifo))]
                except Exception:
                    batch = self.buffer_fifo.sample(len(self.buffer_fifo))
                    rewards = batch["rewards"].flatten()        
                    
            print("rewards", rewards)
            print("current_success", current_success.float().mean().item() * 100.0)
            self.performance = current_success.float().mean().item() * 100.0
            logging.info(f"[share_and_receive] Computed performance: {self.performance} ")
            
            #===========================================================================================
            
            
        except Exception as e:
            logging.error(f"[share_and_receive] Exception before barrier: {e}", exc_info=True)
            if self.barrier:
                try:
                    self.barrier.abort()
                except Exception as be:
                    logging.error(f"[share_and_receive] Failed to abort barrier: {be}", exc_info=True)
            raise

        # --- FORCE SHARING/COMPOSITION FOR TESTING ---
        # Optional: inject fake peers for local testing
        if getattr(self.args, "force_sharing_test", False):
            # Injected peers are put into the shared TEQ list (for this episode)
            test_peer_id = (self.args.agent_id + 1) % self.args.num_agents
            noise = torch.from_numpy(np.random.randn(*self.task_embedding.shape).astype(np.float32)).to(self.task_embedding.device) * 0.01
            peer_emb = (self.task_embedding + noise).cpu().numpy().astype(np.float32)
            peer_perf = max(self.performance - 0.5, 0.1)
            teq_list = self._ensure_episode_list(self.shared_teqs, episode)
            teq_list.append((test_peer_id, peer_emb, peer_perf, episode))
            # a higher-performing peer
            test_peer_id2 = (self.args.agent_id + 2) % self.args.num_agents
            peer_emb2 = (self.task_embedding + noise * 2).cpu().numpy().astype(np.float32)
            peer_perf2 = self.performance + 0.5
            teq_list.append((test_peer_id2, peer_emb2, peer_perf2, episode))
            logging.info(f"[Agent {self.args.agent_id}] Injected fake peers for testing.")


        # --- SHARE TEQ: append to shared_teqs[episode] (broadcast) ---
        teq = (self.args.agent_id, self.task_embedding.cpu().numpy().astype(np.float32), self.performance, episode)
        teq_list = self._ensure_episode_list(self.shared_teqs, episode)
        teq_list.append(teq)
        logging.info(f"[Agent {self.args.agent_id}] Appended TEQ to shared_teqs for episode {episode}")

        # TEQ barrier: wait until all agents have appended TEQs
        logging.debug(f"[Agent {self.args.agent_id}] Waiting on TEQ barrier")
        self.barrier.wait()

        # READ TEQs from the shared list (no draining; everyone reads the same snapshot)
        current_teqs = list(self.shared_teqs.get(episode, []))
        logging.info(f"[Agent {self.args.agent_id}] Read {len(current_teqs)} TEQs for episode {episode}")

        received_teqs = []
        peer_details = []

        for peer_id, peer_emb_np, peer_perf, _ in current_teqs:
            # skip own TEQ
            if peer_id == self.args.agent_id:
                continue

            # compute cosine similarity (numpy)
            # ensure both are normalized
            def normalize(v):
                if type(v) != np.ndarray:
                    v = v.cpu().numpy()
                n = np.linalg.norm(v) + 1e-10
                return v / n


            if self.task_embedding is None:
                # if we somehow did not compute embedding yet, compute now
                self.compute_task_embedding()

            a = normalize(self.task_embedding)
            print(peer_emb_np)
            b = normalize(peer_emb_np)
            cos_sim = float(np.dot(a, b))
            is_similar = cos_sim > self.args.sim_threshold
            is_better = peer_perf > self.performance

            peer_details.append({
                "peer_id": peer_id,
                "peer_perf": peer_perf,
                "cos_sim": cos_sim,
                "is_similar": is_similar,
                "is_better": is_better
            })

            if is_similar:
                received_teqs.append((peer_id, b, peer_perf))

        if peer_details:
            logging.info(f"[Agent {self.args.agent_id}] Peer summary: " +
                         ", ".join([f"id={d['peer_id']} perf={d['peer_perf']:.3f} sim={d['cos_sim']:.3f}"
                                    for d in peer_details]))
        else:
            logging.info(f"[Agent {self.args.agent_id}] No peers to summarize.")

        # For similar peers that are worse, send masks (append to shared_masks[episode])
        serialized_lora = None
        sent_masks = 0
        for peer_id, _, peer_perf in received_teqs:
            if self.performance > peer_perf:
                logging.info(f"[Agent {self.args.agent_id}] I am better than peer {peer_id} (sending mask).")
                if serialized_lora is None:
                    sparse_lora = self.make_sparse_lora()
                    serialized_lora = [p.cpu().numpy().astype(np.float32) for p in sparse_lora]
                mask_list = self._ensure_episode_list(self.shared_masks, episode)
                mask_list.append((self.args.agent_id, serialized_lora, episode))
                sent_masks += 1

        logging.info(f"[Agent {self.args.agent_id}] Sent {sent_masks} masks for episode {episode}")

        # Mask barrier: wait until all agents have appended masks (or not)
        logging.debug(f"[Agent {self.args.agent_id}] Waiting on mask barrier")
        self.barrier.wait()

        # READ masks (everyone reads the same list)
        current_masks = list(self.shared_masks.get(episode, []))
        logging.info(f"[Agent {self.args.agent_id}] Read {len(current_masks)} masks for episode {episode}")

        # Accept masks only from better peers
        better_peer_ids = [p_id for p_id, _, p_perf in received_teqs if p_perf > self.performance]
        logging.info(f"[Agent {self.args.agent_id}] Better peer ids (eligible senders): {better_peer_ids}")

        self.received_masks = {}
        for sender_id, peer_lora_ser, _ in current_masks:
            if sender_id in better_peer_ids:
                # deserialize (they are numpy arrays already)
                peer_lora = [np.array(p, dtype=np.float32) for p in peer_lora_ser]
                self.received_masks[sender_id] = peer_lora
                logging.info(f"[Agent {self.args.agent_id}] Accepted mask from {sender_id}")
            else:
                logging.debug(f"[Agent {self.args.agent_id}] Ignored mask from non-better sender {sender_id}")

        # Update beta weights using self.performance and accepted peers' performances
        peer_id_to_perf = {p_id: p_perf for p_id, _, p_perf in received_teqs}
        total_perf = self.performance
        for peer_id in self.received_masks:
            total_perf += peer_id_to_perf.get(peer_id, 0.0)
        logging.info(f"[Agent {self.args.agent_id}] Calculated total_perf = {total_perf:.6f}")

        if total_perf > 0:
            # reset weights to zero
            if isinstance(self.beta_weights, np.ndarray):
                self.beta_weights.fill(0.0)
            else:  # assume list
                self.beta_weights = [0.0] * len(self.beta_weights)
            self.beta_weights[self.task_idx] = self.performance / (total_perf + 1e-12)
            for peer_id in self.received_masks:
                self.beta_weights[peer_id] = peer_id_to_perf.get(peer_id, 0.0) / (total_perf + 1e-12)
        else:
            logging.warning(f"[Agent {self.args.agent_id}] Total perf <= 0; beta_weights unchanged.")

        logging.info(f"[Agent {self.args.agent_id}] Beta weights updated: {self.beta_weights}]")

        # Compose policy using new weights and masks
        logging.info(f"[Agent {self.args.agent_id}] Starting policy composition")
        self.compose_policy()
        logging.info(f"[Agent {self.args.agent_id}] Finished composition")

        # Final clear barrier: leader deletes per-episode lists to avoid memory growth
        logging.debug(f"[Agent {self.args.agent_id}] Waiting on final clear barrier")
        self.barrier.wait()
        if self.args.agent_id == 0:
            # Leader: cleanup
            if episode in self.shared_teqs:
                del self.shared_teqs[episode]
            if episode in self.shared_masks:
                del self.shared_masks[episode]
            logging.info(f"[Agent {self.args.agent_id}] Leader cleared shared lists for episode {episode}")


        
    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]
        values, actions, logprobs = [], [], []
        for i in range(0, total_batch, self.args.buffer_inferbatch): #just to make sure that we don't run out of memory we just give chunk of observation to model at a time
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
        return (
            torch.cat(values, dim=0).to(device=self.device),
            torch.cat(actions, dim=0).to(device=self.device),
            torch.cat(logprobs, dim=0).to(device=self.device)
        ) #dim=0 means we are concatenating along the first dimension, which is the batch dimension.This stacks the chunks vertically, restoring the full batch.

    def collect(self):
        self.policy.prep_rollout()   #sets to eval mode and disables gradients computations since its not required while rollout(since rollout is similar to eval)
        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)
        return value, action, logprob

    def insert(self, data):
        #.cpu creates a copy in cpu so that numpy can convert it to numpy which is liter to be stored in buffer
        obs_img, actions, logprob, value_preds, rewards, success, done = data #obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)
        obs_img_np = obs_img.cpu().numpy()
        actions_np = actions.to(torch.int32).cpu().numpy()
        logprob_np = logprob.to(torch.float32).cpu().numpy()
        value_preds_np = value_preds.to(torch.float32).cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        
        masks_np = masks.cpu().numpy()
       
        self.buffer.insert(obs_img_np, actions_np, logprob_np, value_preds_np, rewards_np, masks_np)
        
        # Insert into buffer_fifo for embedding/similarity
        # Use only the first environment (or loop if needed)
        # For each env in batch, insert SAR into buffer_fifo
        num_envs = obs_img_np.shape[0]
        for i in range(num_envs):
            # For FIFO buffer, we need obs, action, reward, next_obs, done, info
            # Here, next_obs is not available, so we can use zeros or repeat obs (embedding only needs SAR)
            self.buffer_fifo.insert(
                obs=obs_img_np[i],
                action=actions_np[i],
                reward=rewards_np[i] if rewards_np.ndim == 1 else rewards_np[i][0],
                next_obs=obs_img_np[i],  # Placeholder, not used for embedding
                done=done[i].item() if hasattr(done[i], 'item') else bool(done[i]),
                info=None
            )
        
        print(f"[Runner] Buffer step after insert: {self.buffer.step} (buffer size indicator), obs shape: {self.buffer.obs.shape}, actions shape: {self.buffer.actions.shape}")

    def compute_endup(self):
        #Purpose: compute the value of the final state in the buffer and update the buffer with it.
        #This is commonly used in advantage estimation (e.g., GAE) for actor-critic algorithms.
        self.policy.prep_rollout()
        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()
        self.buffer.endup(next_value)

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
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)
        return info

    @torch.no_grad()
    def eval(self, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])
        obs_img, instruction, info = self.env.reset(obj_set=obj_set)
        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)
            obs_img, reward, done, env_info = self.env.step(action)
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
            obs_img_new, reward, done, env_info = self.env.step(action)
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
        
        
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        
        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            
            # MOSAIC: warmup
            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)
            
            # MOSAIC: rollout
            for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                value, action, logprob = self.collect()
                obs_img, reward, done, env_info = self.env.step(action)
                success = env_info["success"]              # e.g., array([True, False, True])
                
                
                
                data = (obs_img, action, logprob, value, reward, success, done)
                
                self.insert(data)
                
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v
                        
            steps = (episode + 1) * self.args.episode_len * self.args.num_envs
            print(pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))
            
            self.compute_endup()
            del value, action, logprob, obs_img, reward, done
            
            # MOSAIC: Share and receive masks and compute embeddings
            self.share_and_receive(episode, current_success=success)
            
            infos = self.train()
            
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)
                
                
                
            wandb.log(infos, step=steps)
            elapsed_time = time.time() - ep_time
            print(f"{self.args.name}: ep {episode:0>4d} | steps {steps} | e {elapsed_time:.2f}s")
            print(pprint.pformat({k: round(v, 4) for k, v in infos.items()}))
            
            
            # MOSAIC: eval
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
    runner = Runner(args, train_xlsx, test_xlsx, Q_emb, Q_mask)
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