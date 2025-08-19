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
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
import ot  # For Wasserstein embeddings
import pickle
import multiprocessing as mp

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

signal.signal(signal.SIGINT, signal.SIG_DFL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    name: str = "MOSAIC-test"
    num_envs: int = 2 #32
    episode_len: int = 5 #80
    use_same_init: bool = False
    steps_max: int = 2000000
    steps_vh: int = 0
    interval_eval: int = 1
    interval_save: int = 40
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
    num_eval_runs: int = 5
    # MOSAIC-specific args
    comm_interval: int = 5
    agent_id: int = 0
    all_envs: str = ""
    lora_sparsity: float = 0.1  # Top-10% weights kept
    sim_threshold: float = 0.7  # Cosine similarity threshold for mask sharing

class Runner:
    def __init__(self, all_args: Args, Q_emb=None, Q_mask=None):
        self.args = all_args
        self.Q_emb = Q_emb
        self.Q_mask = Q_mask
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
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
        )
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Memory after env init: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        # MOSAIC: Initialize task embedding and performance
        self.task_embedding = None
        self.performance = 0.0  # Mean reward
        self.received_masks = {}  # Dict of {agent_id: LoRA_params}

    def compute_task_embedding(self):
        """Compute Wasserstein Task Embedding from SAR trajectories."""
        # Sample SAR tuples from buffer
        num_samples = 100
        if self.buffer.full:
            obs = self.buffer.obs[:self.buffer.ep_len][:num_samples]
            actions = self.buffer.actions[:self.buffer.ep_len][:num_samples]
            rewards = self.buffer.rewards[:self.buffer.ep_len][:num_samples]
        else:
            obs = self.buffer.obs[:self.buffer.step][:num_samples]
            actions = self.buffer.actions[:self.buffer.step][:num_samples]
            rewards = self.buffer.rewards[:self.buffer.step][:num_samples]
        
        # Flatten batch and env dims: (N, num_env, ...) -> (N*num_env, ...)
        num_env = obs.shape[1]
        obs_flat = obs.reshape(-1, *obs.shape[2:])
        actions_flat = actions.reshape(-1, *actions.shape[2:])
        rewards_flat = rewards.reshape(-1)
        
        # Flatten and normalize (simplified; assumes images flattened)
        states = np.array([o.flatten() for o in obs_flat]) / 255.0
        
        # Handle empty actions array to avoid ValueError
        if actions_flat.size == 0:
            raise ValueError("Actions array is empty in compute_task_embedding. Cannot normalize.")
        actions_flat = actions_flat / (np.max(np.abs(actions_flat)) + 1e-6)
        if rewards_flat.size == 0:
            raise ValueError("Rewards array is empty in compute_task_embedding. Cannot normalize.")
        rewards_flat = rewards_flat / (np.max(np.abs(rewards_flat)) + 1e-6)
        
        sar = np.concatenate([states, actions_flat, rewards_flat[:, None]], axis=1)
        mu_tau = sar / (np.sum(sar, axis=1, keepdims=True) + 1e-6)  # Normalize to distribution
        # Reference distribution (simplified: uniform)
        mu_0 = np.ones_like(mu_tau) / mu_tau.shape[1]
        # Wasserstein distance
        M = ot.dist(mu_tau, mu_0, metric='euclidean')
        v_tau = ot.emd2([], [], M)  # Embedding as Wasserstein vector
        return torch.tensor(v_tau, dtype=torch.float32, device=self.device)

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

    def share_and_receive(self, episode):
        print(f"[share_and_receive] Called at episode {episode}")
        """Broadcast TEQ and receive masks from peers."""
        if episode % self.args.comm_interval == 0:
            print(f"[share_and_receive] Communication interval hit (comm_interval={self.args.comm_interval})")
            # Only compute embedding if buffer has actions
            if not self.buffer.full or self.buffer.actions[:self.buffer.ep_len].size == 0:
                print(f"[share_and_receive] Skipping: buffer has no actions yet. buffer.full={self.buffer.full}, actions.size={self.buffer.actions[:self.buffer.ep_len].size}")
                return
            print("[share_and_receive] Computing task embedding and performance...")
            # Compute embedding and performance
            self.task_embedding = self.compute_task_embedding()
            self.performance = np.mean(self.buffer.rewards)
            print(f"[share_and_receive] Computed embedding (shape: {self.task_embedding.shape}), performance: {self.performance}")
            teq = (self.args.agent_id, self.task_embedding.cpu().numpy(), self.performance)
            self.Q_emb.put(teq)
            print("[share_and_receive] Put TEQ in Q_emb")
            # Serialize sparse LoRA
            sparse_lora = self.make_sparse_lora()
            serialized_lora = [p.detach().cpu().to(torch.float32).numpy() for p in sparse_lora]
            print("[share_and_receive] Serialized sparse LoRA")
            # Listen for TEQs
            self.received_masks = {}
            peer_count = 0
            while not self.Q_emb.empty():
                peer_id, peer_emb, peer_perf = self.Q_emb.get()
                print(f"[share_and_receive] Got peer TEQ: peer_id={peer_id}, peer_perf={peer_perf}")
                if peer_id == self.args.agent_id:
                    print("[share_and_receive] Skipping own TEQ")
                    continue
                cos_sim = torch.cosine_similarity(
                    self.task_embedding, torch.tensor(peer_emb, device=self.device), dim=0
                )
                print(f"[share_and_receive] Cosine similarity with peer {peer_id}: {cos_sim.item():.4f}")
                if cos_sim > self.args.sim_threshold and peer_perf > self.performance:
                    print(f"[share_and_receive] Peer {peer_id} passed sim/perf threshold. Requesting mask.")
                    # Request mask (simulated by waiting for Q_mask)
                    self.Q_mask.put((peer_id, serialized_lora))  # Send own mask as courtesy
                    # Receive mask (assume peer responds)
                    while not self.Q_mask.empty():
                        sender_id, peer_lora = self.Q_mask.get()
                        print(f"[share_and_receive] Got mask from sender_id={sender_id}")
                        if sender_id == peer_id:
                            self.received_masks[sender_id] = [torch.tensor(p, device=self.device) for p in peer_lora]
                            print(f"[share_and_receive] Received and stored mask from peer {peer_id}")
                            peer_count += 1
            print(f"[share_and_receive] Total peers received: {peer_count}")
            # Update beta weights (simplified: reward-guided)
            total_perf = self.performance + sum(self.received_masks.keys(), 0.0)
            print(f"[share_and_receive] total_perf={total_perf}")
            if total_perf > 0:
                self.beta_weights[self.task_idx] = self.performance / (total_perf + 1e-6)
                print(f"[share_and_receive] Updated self beta_weight: {self.beta_weights[self.task_idx]}")
                for peer_id in self.received_masks:
                    peer_idx = self.all_envs.index(ENVIRONMENTS[peer_id]) if peer_id < len(ENVIRONMENTS) else 0
                    self.beta_weights[peer_idx] = self.received_masks[peer_id][0].norm().item() / (total_perf + 1e-6)
                    print(f"[share_and_receive] Updated beta_weight for peer {peer_id}: {self.beta_weights[peer_idx]}")
            print(f"[share_and_receive] Final beta_weights: {self.beta_weights}")
            self.compose_policy()
            print("[share_and_receive] Composed policy with new beta_weights and received masks")

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
        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)
        return value, action, logprob

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)
        obs_img = obs_img.cpu().numpy()
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()
        print("inserting")
        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)
        print("inserted")
        print(f"[Runner] Buffer step after insert: {self.buffer.step} (buffer size indicator), obs shape: {self.buffer.obs.shape}, actions shape: {self.buffer.actions.shape}")

    def compute_endup(self):
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
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)
            for _ in tqdm(range(self.args.episode_len), desc="rollout"):
                value, action, logprob = self.collect()
                obs_img, reward, done, env_info = self.env.step(action)
                data = (obs_img, action, logprob, value, reward, done)
                self.insert(data)
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
                test_eval_runs = [self.eval(obj_set="test") for _ in range(self.args.num_eval_runs)]
                test_mean, test_std = aggregate_eval(test_eval_runs)
                sval_stats = {f"eval/{k}_ood": v for k, v in test_mean.items()}
                sval_stats.update({f"eval/{k}_ood_std": test_std[k] for k in test_mean})
                wandb.log(sval_stats, step=steps)
                print("Test eval mean:", pprint.pformat({k: round(v, 4) for k, v in test_mean.items()}))
                print("Test eval std:", pprint.pformat({k: round(v, 4) for k, v in test_std.items()}))
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)
                self.render(epoch=episode, obj_set="train")
                self.render(epoch=episode, obj_set="test")

def main():
    args = tyro.cli(Args)
    Q_emb = mp.Queue() if args.comm_interval > 0 else None
    Q_mask = mp.Queue() if args.comm_interval > 0 else None
    runner = Runner(args, Q_emb, Q_mask)
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

if __name__ == "__main__":
    main()