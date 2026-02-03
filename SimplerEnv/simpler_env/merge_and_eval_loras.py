#!/usr/bin/env python3
"""
Merge two pretrained LoRA adapters for OpenVLA, then evaluate the merged model
on two tasks.

Example:
    python SimplerEnv/simpler_env/merge_and_eval_loras.py \
        --env-a PutCarrotOnPlateInScene-v1 \
        --env-b StackGreenCubeOnYellowCubeBakedTexInScene-v1 \
        --lora-a-path /path/to/lora_a \
        --lora-b-path /path/to/lora_b \
        --merge-alpha 0.5 \
        --num-eval-runs 3 --num-envs 4 --episode-len 80 --obj-set test
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
import types
from typing import Dict

import numpy as np
import torch
import tyro

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _top_r_svd(mat: torch.Tensor, r: int):
    if mat.numel() == 0 or torch.norm(mat) == 0:
        m, n = mat.shape
        return (
            torch.zeros((m, min(r, n)), device=mat.device, dtype=mat.dtype),
            torch.zeros((min(r, n),), device=mat.device, dtype=mat.dtype),
            torch.zeros((n, min(r, n)), device=mat.device, dtype=mat.dtype),
        )
    mat_for_svd = mat
    if mat.dtype in (torch.float16, torch.bfloat16):
        # torch.linalg.svd CPU does not support bf16/float16; move to float32 for the op
        mat_for_svd = mat.to(dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(mat_for_svd, full_matrices=False)
    r = min(r, S.numel())
    return U[:, :r], S[:r], Vh[:r, :].T


def _project_onto_svd_subspace(X: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
    # U: (m, r), V: (n, r), X: (m, n)
    if U.numel() == 0 or V.numel() == 0:
        return torch.zeros_like(X)
    coeffs = torch.einsum("ir,ij,jr->r", U, X, V)
    proj = torch.zeros_like(X)
    for i in range(U.shape[1]):
        ui = U[:, i].unsqueeze(1)
        vi = V[:, i].unsqueeze(1)
        proj = proj + coeffs[i] * (ui @ vi.T)
    return proj


def _load_lora_state(lora_path: Path) -> Dict[str, torch.Tensor]:
    """Load LoRA adapter weights from a PEFT checkpoint directory."""
    lora_path = Path(lora_path)
    safetensors_path = lora_path / "adapter_model.safetensors"
    bin_path = lora_path / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path))
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find adapter_model.safetensors or adapter_model.bin in {lora_path}"
        )

    return state


def _move_state_to_device(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device) for k, v in state.items()}


def _merge_lora_states(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Dual Orthogonal Projection-inspired merge (dop_merge_simple) for two LoRA adapters.

    Guidance applied: treat `state_a` as both Theta0 (base) and Theta_old, and
    `state_b` as Theta_new. The `alpha` argument is kept for interface
    compatibility but is unused in the iterative merge.

    For 2D tensors (LoRA matrices), apply iterative projections; for others,
    fall back to simple averaging.
    """

    # Hyperparameters from dop_merge_simple
    K = 20
    r = 8
    beta = 0.9
    eta = 1e-2
    clip_eps = 1e-8

    merged: Dict[str, torch.Tensor] = {}
    keys = set(state_a.keys()) | set(state_b.keys())

    for k in keys:
        Wold = state_a.get(k, None)  # Theta_old (also Theta0)
        Wnew = state_b.get(k, None)  # Theta_new

        if Wold is None and Wnew is None:
            continue
        if Wold is None:
            merged[k] = Wnew
            continue
        if Wnew is None:
            merged[k] = Wold
            continue

        # Align dtype/device
        if Wnew.dtype != Wold.dtype:
            Wnew = Wnew.to(dtype=Wold.dtype)
        if Wnew.device != Wold.device:
            Wnew = Wnew.to(device=Wold.device)

        device = Wold.device
        original_dtype = Wold.dtype
        work_dtype = torch.float32 if original_dtype in (torch.float16, torch.bfloat16) else original_dtype

        if Wold.dtype != work_dtype:
            Wold = Wold.to(dtype=work_dtype)
        if Wnew.dtype != work_dtype:
            Wnew = Wnew.to(dtype=work_dtype)
        dtype = original_dtype

        # Theta0 same as Wold per guidance
        W0 = Wold

        if Wold.ndim == 2 and Wnew.ndim == 2 and W0.ndim == 2:
            tau_old = (Wold - W0).detach()
            tau_new = (Wnew - W0).detach()

            Uo, So, Vo = _top_r_svd(tau_old, r)
            Un, Sn, Vn = _top_r_svd(tau_new, r)

            Wstar = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)
            alpha_s_prev = 0.5
            alpha_p_prev = 0.5

            for _ in range(K):
                Wstar.requires_grad_(True)

                delta_old = Wstar - Wold
                delta_new = Wstar - Wnew

                proj_old = _project_onto_svd_subspace(delta_old, Uo, Vo)
                proj_new = _project_onto_svd_subspace(delta_new, Un, Vn)

                Ls = 0.5 * torch.sum((delta_old - proj_old) ** 2)
                Lp = 0.5 * torch.sum((delta_new - proj_new) ** 2)

                grad_Ls = torch.autograd.grad(Ls, Wstar, retain_graph=True, create_graph=False)[0]
                grad_Lp = torch.autograd.grad(Lp, Wstar, retain_graph=False, create_graph=False)[0]

                g_s = grad_Ls.detach().view(-1)
                g_p = grad_Lp.detach().view(-1)
                diff = g_s - g_p
                denom = (diff @ diff).clamp_min(clip_eps)
                numer = ((g_p - g_s) @ g_p)
                alpha_k = float((numer / denom).clamp(0.0, 1.0))

                alpha_s_k = beta * alpha_s_prev + (1.0 - beta) * alpha_k
                alpha_p_k = beta * alpha_p_prev + (1.0 - beta) * (1.0 - alpha_k)

                gk = alpha_s_k * grad_Ls + alpha_p_k * grad_Lp

                with torch.no_grad():
                    Wstar = (Wstar - eta * gk).detach().to(device=device, dtype=dtype)

                alpha_s_prev = alpha_s_k
                alpha_p_prev = alpha_p_k

            merged[k] = Wstar.detach()
        else:
            merged[k] = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

    return merged


@dataclass
class EvalArgs:
    # Tasks
    env_a: str
    env_b: str
    obj_set: str = "test"

    # LoRA inputs and merge control
    lora_a_path: str = ""
    lora_b_path: str = ""
    merge_alpha: float = 0.5  # weight for lora_a; (1 - alpha) for lora_b

    # Model + device
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    device: str = "cuda:0"

    # Eval settings
    seed: int = 0
    num_envs: int = 4
    episode_len: int = 80
    num_eval_runs: int = 4

    # OpenVLA policy settings (kept for compatibility)
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999


def _build_policy(args: EvalArgs) -> OpenVLAPolicy:
    """Instantiate OpenVLAPolicy with LoRA-A loaded for structure, then merge later."""

    device_id = int(args.device.split(":")[-1]) if "cuda" in args.device else 0
    policy_args = types.SimpleNamespace(
        vla_path=args.vla_path,
        vla_unnorm_key=args.vla_unnorm_key,
        vla_load_path=args.lora_a_path,
        vla_lora_rank=args.vla_lora_rank,
        vla_lr=args.vla_lr,
        vla_vhlr=args.vla_vhlr,
        vla_optim_beta1=args.vla_optim_beta1,
        vla_optim_beta2=args.vla_optim_beta2,
        vla_temperature=args.vla_temperature,
        vla_temperature_eval=args.vla_temperature_eval,
    )
    policy = OpenVLAPolicy(policy_args, device_id=device_id)
    policy.vla.eval()
    return policy


def _apply_merged_lora(policy: OpenVLAPolicy, merged_state: Dict[str, torch.Tensor]):
    """Overwrite LoRA weights with merged state."""

    current = policy.vla.state_dict()
    for k, v in merged_state.items():
        current[k] = v
    policy.vla.load_state_dict(current, strict=False)
    policy.vla.eval()


@torch.no_grad()
def _evaluate_env(policy: OpenVLAPolicy, env_id: str, args: EvalArgs) -> Dict[str, float]:
    eval_args = types.SimpleNamespace(
        env_id=env_id,
        num_envs=args.num_envs,
        episode_len=args.episode_len,
        seed=args.seed,
        use_same_init=False,
    )

    unnorm_state = policy.vla.get_action_stats(args.vla_unnorm_key)
    env = SimlerWrapper(eval_args, unnorm_state)

    success_per_run = []
    for run_idx in range(args.num_eval_runs):
        obs_img, instruction, info = env.reset(obj_set=args.obj_set)
        run_success = []
        for _ in range(args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            _, action, _ = policy.get_action(obs, deterministic=True)
            obs_img, reward, done, env_info = env.step(action)
            if "episode" in env_info:
                run_success.extend(env_info["episode"].get("success", []))
        if run_success:
            success_per_run.append(float(np.mean(run_success)))

    mean_success = float(np.mean(success_per_run)) if success_per_run else 0.0
    return {"success_mean": mean_success}


def main():
    args = tyro.cli(EvalArgs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if "cuda" in args.device:
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device} but CUDA is not available.")
        target_device = torch.device(args.device)
    else:
        target_device = torch.device(args.device)

    state_a = _move_state_to_device(_load_lora_state(Path(args.lora_a_path)), target_device)
    state_b = _move_state_to_device(_load_lora_state(Path(args.lora_b_path)), target_device)
    merged_state = _merge_lora_states(state_a, state_b, args.merge_alpha)

    policy = _build_policy(args)
    _apply_merged_lora(policy, merged_state)

    results = {}
    for env_id in [args.env_a, args.env_b]:
        results[env_id] = _evaluate_env(policy, env_id, args)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
