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
import sys
import os
import multiprocessing as mp
import datetime
import pandas as pd
import logging
from pathlib import Path
import tyro
import numpy as np

from train_ms3_ppo import Args, Runner

# List of available environments
ENVIRONMENTS = [
    "PutCarrotOnPlateInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
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

def worker(cli_args, cuda_device, q_emb, q_mask, barrier, manager):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    sys.argv = ["train_ms3_ppo.py"] + cli_args
    
    args = tyro.cli(Args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / timestamp / args.env_id
    log_dir.mkdir(parents=True, exist_ok=True)
    train_xlsx = log_dir / "train.xlsx"
    test_xlsx = log_dir / "test.xlsx"
    pd.DataFrame().to_excel(train_xlsx, index=False)
    pd.DataFrame().to_excel(test_xlsx, index=False)
    log_file = log_dir / "log.txt"
    log_fh = open(log_file, "a")
    sys.stdout = log_fh
    sys.stderr = log_fh
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(log_fh)
        ]
    )
    logging.info("Logging started. Log file: %s", log_file)
    
    runner = Runner(args, train_xlsx, test_xlsx, q_emb, q_mask, barrier, manager)
    print("test")
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
        print("Training started. Log file: %s" % log_file)
        runner.run()
    log_fh.close()

def main():
    args, unknown = parse_args()
    cuda_devices = [d.strip() for d in args.cuda_devices.split(',')]
    hosts = [h.strip() for h in args.hosts.split(',')]
    if args.num_agents > len(ENVIRONMENTS):
        raise ValueError(f"Requested {args.num_agents} agents, but only {len(ENVIRONMENTS)} environments available.")
    if len(cuda_devices) < args.num_agents:
        print(f"Warning: {args.num_agents} agents but only {len(cuda_devices)} CUDA devices. Sharing GPUs.")
    if len(set(hosts)) > 1:
        raise ValueError("Multi-host not supported with shared queues. Set all hosts to localhost.")
    # Create shared queues and barrier for MOSAIC communication
    manager = mp.Manager()
    Q_emb = manager.Queue() if args.comm_interval > 0 else None
    Q_mask = manager.Queue() if args.comm_interval > 0 else None
    barrier = manager.Barrier(args.num_agents) if args.comm_interval > 0 else None
    from multiprocessing import Manager
    manager = Manager()
    shared_teqs = manager.dict()   # episode -> manager.list()
    shared_masks = manager.dict()  # episode -> manager.list()

    procs = []
    for i in range(args.num_agents):
        device = cuda_devices[i % len(cuda_devices)]
        agent_name = f"{args.base_name}_{i}"
        agent_seed = args.base_seed + i
        env_id = ENVIRONMENTS[i]
        agent_cli_args = [
            "--env-id", env_id,
            "--name", agent_name,
            "--seed", str(agent_seed),
            "--comm-interval", str(args.comm_interval),
            "--agent-id", str(i),
            "--all-envs", ",".join(ENVIRONMENTS[:args.num_agents]),
           
        ] + unknown
        print(f"Launching agent {i}: name={agent_name}, seed={agent_seed}, env_id={env_id}, CUDA_VISIBLE_DEVICES={device}")
        proc = mp.Process(target=worker, args=(agent_cli_args, device, shared_teqs, shared_masks, barrier, manager))
        proc.start()
        procs.append(proc)

    # Wait for all processes to finish
    for i, proc in enumerate(procs):
        proc.join()
        print(f"Agent {i} (PID {proc.pid}) exited with code {proc.exitcode}")

if __name__ == "__main__":
    main()