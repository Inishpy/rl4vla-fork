#!/usr/bin/env python3
"""
Multi-agent VLA training launcher.

Usage:
    python train_multi_vla.py --num_agents 2 --cuda_devices 0,1 --other_args ...

Arguments:
    --num_agents: Number of parallel VLA training pipelines to launch.
    --cuda_devices: Comma-separated list of CUDA device ids to assign (e.g., "0,1,2").
    All other arguments are passed through to train_ms3_ppo.py.

Each agent will be launched as a separate process, with its own CUDA device and unique name/seed.
"""

import argparse
import subprocess
import sys
import os

# List of available environments (edit as needed)
ENVIRONMENTS = [
    "PutCarrotOnPlateInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent VLA training launcher", allow_abbrev=False)
    parser.add_argument('--num_agents', type=int, required=True, help='Number of agents to launch')
    parser.add_argument('--cuda_devices', type=str, required=True, help='Comma-separated CUDA device ids (e.g., "0,1,2")')
    parser.add_argument('--hosts', type=str, default="localhost", help='Comma-separated list of hosts (e.g., "localhost,server2,server3")')
    parser.add_argument('--base_seed', type=int, default=0, help='Base seed for agents (each agent gets base_seed + i)')
    parser.add_argument('--base_name', type=str, default="PPO-multi", help='Base name for runs (each agent gets base_name_i)')
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown = parse_args()
    cuda_devices = [d.strip() for d in args.cuda_devices.split(',')]
    hosts = [h.strip() for h in args.hosts.split(',')]
    # if len(cuda_devices) < args.num_agents:
    #     raise ValueError(f"Not enough CUDA devices ({len(cuda_devices)}) for {args.num_agents} agents.")

    procs = []
    for i in range(args.num_agents):
        device = cuda_devices[i % len(cuda_devices)]
        agent_name = f"{args.base_name}_{i}"
        agent_seed = args.base_seed + i
        env_id = ENVIRONMENTS[i % len(ENVIRONMENTS)]
        host = hosts[i % len(hosts)]

        # Build command for train_ms3_ppo.py
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "train_ms3_ppo.py"),
            "--name", agent_name,
            "--seed", str(agent_seed),
            "--env_id", env_id,
        ] + unknown

        # Set CUDA_VISIBLE_DEVICES for this process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device

        if host == "localhost" or host == "127.0.0.1" or host == os.uname()[1]:
            print(f"Launching agent {i} locally: name={agent_name}, seed={agent_seed}, env_id={env_id}, CUDA_VISIBLE_DEVICES={device}")
            proc = subprocess.Popen(cmd, env=env)
        else:
            # Build remote command string
            remote_cmd = (
                f"CUDA_VISIBLE_DEVICES={device} "
                f"{sys.executable} {os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_ms3_ppo.py'))} "
                f"--name {agent_name} --seed {agent_seed} --env_id {env_id} " +
                " ".join(unknown)
            )
            print(f"Launching agent {i} on {host}: name={agent_name}, seed={agent_seed}, env_id={env_id}, CUDA_VISIBLE_DEVICES={device}")
            # Use SSH to launch the process remotely
            proc = subprocess.Popen(
                ["ssh", host, remote_cmd],
                env=env
            )
        procs.append(proc)

    # Wait for all processes to finish
    for i, proc in enumerate(procs):
        ret = proc.wait()
        print(f"Agent {i} (PID {proc.pid}) exited with code {ret}")

if __name__ == "__main__":
    main()