#!/usr/bin/env python3
"""
Multi-agent VLA training launcher with MOSAIC collaboration.

Usage:
    python train_multi_vla.py --num_agents 4 --cuda_devices 0,1,2,3 --other_args ...

Arguments:
    --num_agents: Number of parallel VLA training pipelines to launch.
    --cuda_devices: Comma-separated list of CUDA device ids (e.g., "0,1,2,3").
    All other arguments are passed through to train_ms3_ppo.py.
"""

import argparse
import subprocess
import sys
import os
import multiprocessing as mp

# List of available environments
ENVIRONMENTS = [
    "PutCarrotOnPlateInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
    "PutOnPlateInScene25Single-v1",
    "PutOnPlateInScene25Main-v3",
    "PutOnPlateInScene25VisionImage-v1",
    "PutOnPlateInScene25VisionTexture03-v1",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent VLA training launcher with MOSAIC", allow_abbrev=False)
    parser.add_argument('--num_agents', type=int, required=True, help='Number of agents to launch (max: %d)' % len(ENVIRONMENTS))
    parser.add_argument('--cuda_devices', type=str, required=True, help='Comma-separated CUDA device ids (e.g., "0,1,2,3")')
    parser.add_argument('--hosts', type=str, default="localhost", help='Comma-separated list of hosts (e.g., "localhost,server2")')
    parser.add_argument('--base_seed', type=int, default=0, help='Base seed for agents (each gets base_seed + i)')
    parser.add_argument('--base_name', type=str, default="vla-multi", help='Base name for runs (each gets base_name_i)')
    parser.add_argument('--comm_interval', type=int, default=5, help='Episodes between communication rounds')
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown = parse_args()
    cuda_devices = [d.strip() for d in args.cuda_devices.split(',')]
    hosts = [h.strip() for h in args.hosts.split(',')]
    if args.num_agents > len(ENVIRONMENTS):
        raise ValueError(f"Requested {args.num_agents} agents, but only {len(ENVIRONMENTS)} environments available.")
    if len(cuda_devices) < args.num_agents and len(set(hosts)) == 1:
        print(f"Warning: {args.num_agents} agents but only {len(cuda_devices)} CUDA devices on {hosts[0]}. Sharing GPUs.")

    # Create shared queues for MOSAIC communication
    manager = mp.Manager()
    Q_emb = manager.Queue()  # Task embeddings (TEQ: v_tau, r_tau, agent_id)
    Q_mask = manager.Queue()  # Mask responses (agent_id, serialized LoRA)

    procs = []
    for i in range(args.num_agents):
        device = cuda_devices[i % len(cuda_devices)]
        agent_name = f"{args.base_name}_{i}"
        agent_seed = args.base_seed + i
        env_id = ENVIRONMENTS[i]
        host = hosts[i % len(hosts)]

        # Pass all environments and agent ID for similarity computation
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "train_ms3_ppo.py"),
            "--name", agent_name,
            "--seed", str(agent_seed),
            "--env_id", env_id,
            "--comm_interval", str(args.comm_interval),
            "--agent_id", str(i),
            "--all_envs", ",".join(ENVIRONMENTS[:args.num_agents]),  # Pass subset for this run
        ] + unknown

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device

        if host == "localhost" or host == "127.0.0.1" or host == os.uname()[1]:
            print(f"Launching agent {i} locally: name={agent_name}, seed={agent_seed}, env_id={env_id}, CUDA_VISIBLE_DEVICES={device}")
            proc = subprocess.Popen(cmd, env=env)
        else:
            remote_cmd = (
                f"CUDA_VISIBLE_DEVICES={device} "
                f"{sys.executable} {os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_ms3_ppo.py'))} "
                f"--name {agent_name} --seed {agent_seed} --env_id {env_id} "
                f"--comm_interval {args.comm_interval} --agent_id {i} --all_envs {','.join(ENVIRONMENTS[:args.num_agents])} "
                + " ".join(unknown)
            )
            print(f"Launching agent {i} on {host}: name={agent_name}, seed={agent_seed}, env_id={env_id}, CUDA_VISIBLE_DEVICES={device}")
            proc = subprocess.Popen(["ssh", host, remote_cmd], env=env)
        procs.append(proc)

    # Wait for all processes to finish
    for i, proc in enumerate(procs):
        ret = proc.wait()
        print(f"Agent {i} (PID {proc.pid}) exited with code {ret}")

if __name__ == "__main__":
    main()