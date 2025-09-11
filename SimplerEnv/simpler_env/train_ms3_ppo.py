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
from torch.utils.data import DataLoader, SubsetRandomSampler 
import tempfile
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
from simpler_env.communication import communicate
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
from dotenv import load_dotenv
import wandb

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
hf_key = os.getenv("HUGGINGFACE_TOKEN")
# Login to Weights & Biases
wandb.login(key=api_key)
from huggingface_hub import login
login(token=hf_key)

from simpler_env.helpers import   obs_to_clip_features, obs_to_vit_features, obs_to_cnn_features, to_tensor_on_device, safe_compute_f, obs_to_dinov3_features

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
    
    num_envs: int = 16
    episode_len: int = 80
    use_same_init: bool = False
    steps_max: int = 2000000
    steps_vh: int = 0
    interval_eval: int = 2
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
    wandb: bool = False
    only_render: bool = False
    render_info: bool = False
    num_eval_runs: int = 1
    # MOSAIC-specific args
    force_sharing_test: bool = False
    comm_interval: int = 3
    comm_start: int = 7
    agent_id: int = 0
    all_envs: str = ""
    
    lora_sparsity: float = 1.0  # Top-10% weights kept
    sim_threshold: float = 0.7  # Cosine similarity threshold for mask sharing

class Runner:
    def __init__(self, all_args: Args, train_xlsx=None, test_xlsx=None,sim_dir=None, param_drift_dir=None, Q_emb=None, Q_mask=None):
        self.args = all_args
        self.train_xlsx = train_xlsx  # Store log directory for Excel output
        self.test_xlsx = test_xlsx
        self.sim_dir = sim_dir
        self.param_drift_dir = param_drift_dir
        self.q_emb = Q_emb
        self.q_mask = Q_mask
        # self.shared_teqs = shared_teqs
        # self.shared_masks = shared_masks
        # self.manager = manager
        # self.barrier = barrier
        
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
        # self.beta_weights is now handled as a trainable parameter in the policy

        self.alg = OpenVLAPPO(all_args, self.policy)
        unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        
        self.env = SimlerWrapper(self.args, unnorm_state)
        
        #self.action_space = getattr(self.env.action_space, 'shape', None)
        # print(f"Action space: {self.action_space}")
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

        self.ref = None
        # Set reference for embeddings (self.ref)
        self.set_reference(
            a_task_observation_dim=int(512),
            some_reference_num=128,
            some_action_dim=7
        )

        # MOSAIC: Initialize task embedding and performance
        self.task_embedding = None
        self.performance = 0.0  # Mean reward
        self.received_masks = {}  # Dict of {agent_id: LoRA_params}


    def compute_task_embedding(self, num_samples=128, M_ref=None, use_sinkhorn=True, sinkhorn_reg=1e-2, debug=True):
        """
        Improved compute_task_embedding with feature standardization, stable sinkhorn attempt,
        and diagnostics to show why all-zero embeddings may occur.

        Returns: torch.Tensor flattened (M_ref * d,)
        """
        sample = self.buffer_fifo.obs[0].shape
        sample = np.asarray(sample)
        input_dim = int(np.prod(sample.shape))
        
        # 1) sample
        num_available = len(self.buffer_fifo)
        num_samples = min(num_samples, num_available)
        if num_samples == 0:
            raise ValueError("buffer_fifo is empty in compute_task_embedding. Cannot compute embedding.")

        idxs = np.random.choice(num_available, num_samples, replace=False)
        obs = self.buffer_fifo.obs[idxs]
        actions = self.buffer_fifo.actions[idxs]
        rewards = self.buffer_fifo.rewards[idxs]

        
        # Suppose your images are in a list/array called `obs`
        obs_array = np.array(obs, dtype=object)  # numpy array of images

        # Create temporary files to pass input/output
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_in, \
            tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_out:
            input_path = Path(tmp_in.name)
            output_path = Path(tmp_out.name)
            np.save(input_path, obs_array)

        # Run the worker script in a subprocess
        # You can activate a specific environment using conda run or your Python path
        """
        result = subprocess.run(
            [
                "conda", "run", "-n", "dinov3", "python", "simpler_env/dinov3_worker.py",
                str(input_path),
                str(output_path)
            ],
            capture_output=True,
            text=True
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        # Load the result back
        states = np.load(output_path)
        print("Shape of features:", states.shape)
        """
        # 2) build X (N x d)
        logging.info("[compute_task_embedding] obs shape: %s", obs.shape)
        states = obs_to_cnn_features(obs, device='cuda')  # (N, 512)#np.array([o.flatten() for o in obs], dtype=np.float64) / 255.0  # (N, ds)
        # 1) CLIP embeddings (recommended for multimodal LLM compatibility)
        #states = obs_to_vit_features(obs)
        # states= obs_to_dinov3_features(obs)
        # states = obs_to_clip_features(obs, model_name='openai/clip-vit-base-patch32', batch_size=8)
        # X_clip.shape -> (N, 512) typically
        logging.info("[compute_task_embedding] states shape: %s", states.shape)
        if actions.size == 0:
            raise ValueError("Actions array is empty in compute_task_embedding. Cannot normalize.")
        actions = actions.astype(np.float64)
        max_act = np.max(np.abs(actions)) + 1e-9
        actions_flat = actions / max_act
        if actions_flat.ndim == 1:
            actions_flat = actions_flat[:, None]  # ensure shape (N, da)

        if rewards.size == 0:
            raise ValueError("Rewards array is empty in compute_task_embedding. Cannot normalize.")
        rewards = rewards.astype(np.float64)
        max_r = np.max(np.abs(rewards)) + 1e-9
        rewards_flat = (rewards / max_r)[:, None]

        X = np.concatenate([states, actions_flat, rewards_flat], axis=1)  # (N, d)
        logging.info("[compute_task_embedding] X shape: %s", X.shape)
        X_ = self.lwe(X, (2,7), input_dim)
        print("X_ shape:", X_.shape, "X shape:", X.shape)
        #final_vec = np.concatenate([np.empty(10), X.ravel()], axis=0)
        final_tensor = torch.from_numpy(X.ravel()).float().to(X_.device)

        return final_tensor
        

    
    def preprocess_dataset(self, X, some_task_action_space_size, input_dim):
        '''Function that preprocess the Data-Batch of SAR before calcuting the embedding'''
        logging.info("Preprocessing SAR")
        if True: #self.num_samples is not None and len(X) > self.num_samples:
            logging.info(f"len(X): {len(X)}")
            idxs = np.sort(np.random.choice(len(X), 10, replace=False))
            sampler = SubsetRandomSampler(idxs)
            loader = DataLoader(X, sampler=sampler, batch_size=64)
            logging.info(f"1len(loader): {len(loader)}")
        else:
            ## No subsampling
            logging.info(f"len(X): {len(X)}")
            loader = DataLoader(X, batch_size=64)
        logging.info(f"2len(loader): {len(loader)}")  
        X = []
        q=0
        for batch in loader:
            q = q+1
            X.append(batch.squeeze().view(batch.shape[0],-1))
        X = torch.cat(X).to(self.device)
        #print("QQQQQQQQQQQQ:", q)
        #print("FIRST X:", X)
        logging.info(f"X.shape: {X.shape}")
        img = X[:,:input_dim]
        act = X[:,input_dim:-1]
        reward = X[:,-1].unsqueeze(1)
        logging.info(f"img.shape: {img.shape}")
        if True: #self.normalized:
            mean = torch.mean(img.float())
            std = torch.std(img.float())
            img = (img.float()-mean)/std
            
        logging.info("Preprocessing done")
        if True: #self.oh:
            act_oh = torch.zeros(X.shape[0], some_task_action_space_size)
            logging.info(f"act_oh.shape: {act_oh.shape}")
            for i in range(act.shape[0]):
                act_oh [i,int(act[i])]=1
            act = act_oh.to(self.device)
            #print("act_OH:", act_oh)
            #lb = preprocessing.LabelBinarizer()
            #lb.fit(act_.cpu())
            #act = lb.transform(act_.cpu())
        logging.info(f"act.shape: {act.shape}")
        return torch.cat((img, act, reward), dim=1).float()
        # return X.float()

    def lwe(self, X, some_task_action_space_size, input_dim):
        '''Calculates the Embedding for a given Data-Batch of SAR
        Returns a 1D Tensor with the calculated Embedding'''
        logging.info("Calculating Embedding")
        #
        #print("PrePRocess_SAR_DETECT:", X)
        #print("What we actually use form the data we give:", X.shape)
        logging.info(self.ref.shape)
        ref_size = self.ref.shape[0]
        logging.info(self.ref.shape)
        print("X:", X.shape, "ref:", self.ref.shape)
        self.set_reference(
            a_task_observation_dim=X.shape[1]-8,
            some_reference_num=128,
            some_action_dim=7
        )
        print("X:", X.shape, "ref:", self.ref.shape)
        #C = ot.dist(X.cpu(), self.ref).cpu().numpy()
        # Make sure both are tensors on the same device
        # X_t = X.to("cuda")          # or stay on CPU if GPU memory is limited
        # ref_t = self.ref.to("cuda")
        # logging.info("Preprocessing SAR done")
        # # Compute pairwise squared Euclidean distances
        # C = torch.cdist(X_t, ref_t, p=2)   # shape (12, 50)
        logging.info("Calculating Embedding done")
        # convert/slice both (handles X as numpy or tensor, and self.ref as torch tensor)
        X_t   = to_tensor_on_device(X[:, :], self.device)        # (12, 10)
        ref_t = to_tensor_on_device(self.ref[:, :], self.device) # (50, 10)

        # sanity checks
        assert X_t.shape[1] == ref_t.shape[1], "feature dims must match"
        assert X_t.dtype == torch.float32 and ref_t.dtype == torch.float32

        # compute distances on device
        C_t = torch.cdist(X_t, ref_t)     # shape (12, 50)

        # bring to cpu numpy if you need it in numpy
        C = C_t.cpu().numpy()
        print("C.shape", C.shape)
        # # If you need a numpy array
        # C = C.cpu().numpy()
        #C = ot.dist(X[:,:10].cpu(), self.ref[:,:10].cpu()).numpy()
        logging.info(self.ref.shape)
        # Calculating the transport plan
        gamma = torch.from_numpy(ot.emd(ot.unif(X.shape[0]), ot.unif(ref_size), C, numItermax=700000)).float()
        # Calculating the transport map via barycenter projection /gamma.sum(dim=0).unsqueeze(1)
        logging.info(self.ref.shape)
        f= safe_compute_f(ref_size, gamma, X, self.ref, device=torch.device("cuda"), chunk_cols=1024)#(torch.matmul((ref_size*gamma).T,X.cpu())-self.ref)/np.sqrt(ref_size)
        logging.info(self.ref.shape)
        
        return f.ravel()

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
        # Use beta_weights from policy (trainable tensor)
        beta_weights = self.policy.beta_weights
        for i, param in enumerate(own_lora):
            weighted_sum = param * beta_weights[self.task_idx]
            for agent_id, peer_lora in self.received_masks.items():
                peer_idx = self.all_envs.index(ENVIRONMENTS[agent_id]) if agent_id < len(ENVIRONMENTS) else 0
                weighted_sum += peer_lora[i] * beta_weights[peer_idx]
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
        
    
    def share_and_receive(self, episode, current_success, timeout=60.0):
        """
        Communication using local queues (self.q_emb, self.q_mask) and the TCP comm module.

        Expected:
        - self.q_emb is a multiprocessing.Queue() consumed by communication_module.
            We put (vi_list, performance, mask_id, serialized_mask) into it.
        - self.q_mask is a multiprocessing.Queue() filled by communication_module with
            tuples (peer_agent_id, peer_perf, peer_mask_ser).
        """
        logging.info(f"[share_and_receive] Called at episode {episode}")

        # --- Only compute embedding if FIFO has data (like original) ---
        if len(self.buffer_fifo) > 0:
            logging.info("[share_and_receive] FIFO buffer has data; proceeding to compute embedding and performance")
            self.task_embedding = self.compute_task_embedding()
            
            print(self.task_embedding)
            logging.info(f"[share_and_receive] Task embedding computed: shape={getattr(self.task_embedding, 'shape', None)}")
            # --- Save embedding for similarity heatmap ---
            try:
                
                emb_path = self.sim_dir / f"embedding_agent_{self.args.agent_id}_ep_{episode}.npy"
                np.save(emb_path, self.task_embedding.cpu().numpy())
                logging.info(f"[share_and_receive] Saved embedding to {emb_path}")
            except Exception as e:
                logging.error(f"[share_and_receive] Failed to save embedding for heatmap: {e}", exc_info=True)
        
        
        communicate(self, episode, current_success, timeout)
        

    
        
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

    def set_reference(self, a_task_observation_dim, some_reference_num, some_action_dim):
        '''A setter method, for manually setting and updating the reference for calculating
        the tasks embeddings.'''
        torch.manual_seed(98)
        reference = torch.rand(
            some_reference_num,
            a_task_observation_dim + some_action_dim + 1
        )  # Plus one which is the reward.
        self.ref = reference.to(self.device)

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

            # --- Save policy checkpoint for parameter drift ---
            try:
                from safetensors.torch import save_file as safetensors_save_file
                # Compose parameter drift directory path
                print("drift_dir:", self.param_drift_dir)
                drift_dir = self.param_drift_dir
                drift_dir.mkdir(parents=True, exist_ok=True)
                drift_path = drift_dir / f"policy_agent_{self.args.agent_id}_ep_{episode}.safetensors"
                safetensors_save_file(self.policy.vla.state_dict(), str(drift_path))
                print(f"[Runner] Saved parameter drift checkpoint: {drift_path}")
            except Exception as e:
                print(f"[Runner] Failed to save parameter drift checkpoint: {e}")

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
                self.policy.save(save_path)

                self.render(epoch=episode, obj_set="train")
                self.render(epoch=episode, obj_set="test")

def main():
    args = tyro.cli(Args)
    # --- Logging and logdir setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / timestamp / args.env_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "@similarityheat").mkdir(parents=True, exist_ok=True)
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