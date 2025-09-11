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

# Set this variable to True to log only to file (and create log folders), or False to log only to terminal.
LOG_TO_FILE_ONLY = True

# Set this variable to True to enable multiprocessing and server sockets,
# or False to run a single VLA agent without multiprocessing or communication.
multiprocess = True




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
import socket
import json
import numpy as np
import time
# mp.set_start_method('spawn', force=True)

from train_ms3_ppo import Args, Runner
import torch
from simpler_env.visualize import generate_similarity_heatmap
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



# Helper functions for TCP messaging
def send_message(conn, msg):
    data = json.dumps(msg).encode('utf-8')
    conn.sendall(len(data).to_bytes(4, 'big'))
    conn.sendall(data)

def receive_message(conn):
    len_bytes = b''
    while len(len_bytes) < 4:
        packet = conn.recv(4 - len(len_bytes))
        if not packet:
            return None
        len_bytes += packet
    length = int.from_bytes(len_bytes, 'big')
    data = b''
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return json.loads(data.decode('utf-8'))

def send_to_addr(target_addr, msg):
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect(target_addr)
    send_message(conn, msg)
    conn.close()

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Algorithm 5: Communication Server (modified to include agent_id and perf)
def communication_server(addr_i, W, server_side_selection, sim_threshold, own_data, lock, query_responses, acquired_masks, running_flag, agent_id):
    own_data['agent_id'] = agent_id  # Set here too for safety
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(addr_i)
    server.listen(W)
    logging.info(f"[Server] Listening on {addr_i}")

    while running_flag.is_set():
        try:
            conn, remote_addr = server.accept()
            message = receive_message(conn)
            conn.close()
            if message is None:
                continue

            msg_type = message.get('type')

            if msg_type == 'TEQ':
                vi = message['vi']
                ri = message['ri']
                addri = tuple(message['addri'])

                with lock:
                    vj = own_data['v']
                    rj = own_data['r']
                    mask_id = own_data['mask_id']
                    mask = own_data['mask']
                    addrj = own_data['addr']
                    agent_id_j = own_data['agent_id']

                if vj is None or rj is None:
                    continue

                cos_sim_val = cosine_sim(vi, vj)
                is_similar = cos_sim_val > sim_threshold
                is_better = rj > ri

                if server_side_selection:
                    if is_similar and is_better:
                        mtr = {'type': 'MTR', 'mask': mask, 'rj': rj, 'agent_id': agent_id_j}
                        send_to_addr(addri, mtr)
                else:
                    qr = {'type': 'QR', 'vj': vj, 'rj': rj, 'mask_id': mask_id, 'addrj': list(addrj), 'agent_id': agent_id_j}
                    send_to_addr(addri, qr)

            elif msg_type == 'QR':
                vj = message['vj']
                rj = message['rj']
                mask_id = message['mask_id']
                addrj = tuple(message['addrj'])
                agent_id_j = message['agent_id']
                with lock:
                    query_responses.append((vj, rj, mask_id, addrj, agent_id_j))

            elif msg_type == 'MR':
                mask_id_req = message['mask_id']
                addri = tuple(message['addri'])
                with lock:
                    if mask_id_req == own_data['mask_id']:
                        mtr = {'type': 'MTR', 'mask': own_data['mask'], 'rj': own_data['r'], 'agent_id': own_data['agent_id']}
                        send_to_addr(addri, mtr)

            elif msg_type == 'MTR':
                mask = message['mask']
                rj = message['rj']
                agent_id_j = message['agent_id']
                with lock:
                    acquired_masks.append((agent_id_j, rj, mask))

        except Exception as e:
            logging.error(f"[Server] Error: {e}")

# Algorithm 3: MOSAIC Communication Module (modified)
def communication_module(Q_emb, Q_mask, addr_i, peer_addrs, server_side_selection, sim_threshold, agent_id, timeout=60.0):
    manager = mp.Manager()
    own_data = manager.dict({'v': None, 'r': None, 'mask_id': None, 'mask': None, 'addr': addr_i, 'agent_id': agent_id})
    lock = manager.Lock()
    query_responses = manager.list()
    acquired_masks = manager.list()
    running_flag = manager.Event()
    running_flag.set()

    # Start server process
    server_proc = mp.Process(target=communication_server, args=(addr_i, len(peer_addrs) + 1, server_side_selection, sim_threshold, own_data, lock, query_responses, acquired_masks, running_flag, agent_id))
    server_proc.start()

    try:
        while True:
            # Wait for query from Q_emb (vi as list, ri as float)
            vi, ri = Q_emb.get()

            # Compute mask here in Runner, but for comm, assume put after computing
            # In Runner/share_and_receive, compute mask if needed, but here assume own_mask is set
            # Wait, in code, set own_data after get

            # But to set mask, need to know when to compute
            # Assume Runner puts (vi, ri, mask_id, mask_ser)
            # Wait, to make it work, modify to Q_emb.get() -> vi, ri, mask_id, mask_ser

            # For now, assume dummy, but in actual, change Q_emb.put to include mask_id, mask_ser
            mask_id = 'current'
            mask = [[1.0, 2.0]]  # Dummy, replace in actual with serialized LoRA

            with lock:
                own_data['v'] = vi
                own_data['r'] = ri
                own_data['mask_id'] = mask_id
                own_data['mask'] = mask

            if server_side_selection:
                with lock:
                    acquired_masks[:] = []

                teq = {'type': 'TEQ', 'vi': vi, 'ri': ri, 'addri': list(addr_i)}
                for peer_addr in peer_addrs:
                    send_to_addr(peer_addr, teq)

                start = time.time()
                while time.time() < start + timeout:
                    with lock:
                        if acquired_masks:
                            break
                    time.sleep(0.1)

                with lock:
                    for item in acquired_masks:
                        Q_mask.put(item)  # (agent_id_j, rj, mask)

            else:
                with lock:
                    query_responses[:] = []

                teq = {'type': 'TEQ', 'vi': vi, 'ri': ri, 'addri': list(addr_i)}
                for peer_addr in peer_addrs:
                    send_to_addr(peer_addr, teq)

                start = time.time()
                while time.time() < start + timeout:
                    with lock:
                        if len(query_responses) >= len(peer_addrs):
                            break
                    time.sleep(0.1)

                with lock:
                    qrs = list(query_responses)

                P = []
                for vj, rj, mask_id_j, addrj, agent_id_j in qrs:
                    cos_sim_val = cosine_sim(vi, vj)
                    is_similar = cos_sim_val > sim_threshold
                    is_better = rj > ri
                    if is_similar and is_better:
                        P.append((addrj, mask_id_j, rj, agent_id_j))

                with lock:
                    acquired_masks[:] = []

                for addrj, mask_id_j, rj, agent_id_j in P:
                    mr = {'type': 'MR', 'mask_id': mask_id_j, 'addri': list(addr_i)}
                    send_to_addr(addrj, mr)

                start = time.time()
                while time.time() < start + timeout:
                    with lock:
                        if len(acquired_masks) >= len(P):
                            break
                    time.sleep(0.1)

                with lock:
                    masks_items = list(acquired_masks)
                    # Since order may not match, but for simplicity, assume they do; alternatively, log and use
                    for item in masks_items:
                        Q_mask.put(item)  # (agent_id_j, rj, mask)

    except KeyboardInterrupt:
        running_flag.clear()
        server_proc.join()

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent VLA training launcher with MOSAIC", allow_abbrev=False)
    parser.add_argument('--num_agents', type=int, required=True, help='Number of agents to launch (max: %d)' % len(ENVIRONMENTS))
    parser.add_argument('--cuda_devices', type=str, required=True, help='Comma-separated CUDA device ids (e.g., "0,1,2,3")')
    parser.add_argument('--hosts', type=str, default="localhost", help='Comma-separated list of hosts (e.g., "localhost,server2")')
    parser.add_argument('--base_port', type=int, default=5000, help='Base port for TCP communication')
    parser.add_argument('--base_seed', type=int, default=0, help='Base seed for agents (each gets base_seed + i)')
    parser.add_argument('--base_name', type=str, default="vla-multi", help='Base name for runs (each gets base_name_i)')
    parser.add_argument('--comm_interval', type=int, default=5, help='Episodes between communication rounds')
    parser.add_argument('--server_side_selection', action='store_true', help='Enable server-side peer selection (default: client-side)')
    parser.add_argument('--sim_threshold', type=float, default=0.8, help='Cosine similarity threshold for task alignment')
    args, unknown = parser.parse_known_args()
    return args, unknown

def worker(cli_args, timestamp, cuda_device, addr_i, peer_addrs, server_side_selection, sim_threshold):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["PARAM_DRIFT_TIMESTAMP"] = timestamp  # Ensure subprocesses get the correct timestamp
    sys.argv = ["train_ms3_ppo.py"] + cli_args

    args = tyro.cli(Args)

    if LOG_TO_FILE_ONLY:
        log_dir = Path("logs") / timestamp / args.env_id
        log_dir.mkdir(parents=True, exist_ok=True)

        sim_dir = Path("logs") / timestamp / "similarityheat"
        pram_drift_dir = Path("logs") / timestamp / "parameter_drift"
        sim_dir.mkdir(parents=True, exist_ok=True)
        pram_drift_dir.mkdir(parents=True, exist_ok=True)
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
    else:
        # Terminal logging only: do not create log folders/files, do not redirect stdout/stderr
        log_dir = None
        sim_dir = None
        pram_drift_dir = None
        train_xlsx = None
        test_xlsx = None
        log_fh = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging started. Outputting to terminal only.")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # # enforce small thread pools in the child too
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    
    # torch.set_num_threads(1)

    # Now safe to parse args & import heavy libs if needed
    sys.argv = ["train_ms3_ppo.py"] + cli_args
   
    args = tyro.cli(Args)

    # Logging code unchanged...
    # -------------------------------------------------------------------
    # Start communication module using spawn context too:
    
    Q_emb = mp.Queue()
    Q_mask = mp.Queue()

    # Start communication module
    comm_proc = mp.Process(target=communication_module, args=(Q_emb, Q_mask, addr_i, peer_addrs, server_side_selection, sim_threshold, args.agent_id))
    comm_proc.start()

    runner = Runner(args, train_xlsx, test_xlsx, sim_dir, pram_drift_dir,Q_emb, Q_mask)
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
        if LOG_TO_FILE_ONLY:
            print("Training started. Log file: %s" % log_file)
        else:
            print("Training started. Logging to terminal only.")
        runner.run()
    if LOG_TO_FILE_ONLY and log_fh is not None:
        log_fh.close()

    # Cleanup
    comm_proc.terminate()
    comm_proc.join()

import threading

def main(timestamp):
    args, unknown = parse_args()
    global multiprocess
    # call it early in main() before spawning processes:
    #predownload_timm('vit_base_patch16_224', pretrained=True)
    if not multiprocess:
        # Single-agent, no multiprocessing, no server sockets, no communication
        # Use only the first environment, first CUDA device, etc.
        cuda_devices = [d.strip() for d in args.cuda_devices.split(',')]
        device = cuda_devices[0] if cuda_devices else "0"
        env_id = ENVIRONMENTS[0]
        agent_name = f"{args.base_name}_0"
        agent_seed = args.base_seed
        # Compose CLI args for single agent
        agent_cli_args = [
            "--env-id", env_id,
            "--name", agent_name,
            "--seed", str(agent_seed),
            "--comm-interval", str(args.comm_interval),
            "--agent-id", "0",
            "--all-envs", env_id,
        ] + unknown

        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        sys.argv = ["train_ms3_ppo.py"] + agent_cli_args

        # Parse Args as in worker
        agent_args = tyro.cli(Args)

        # Logging setup (copied from worker)
        if LOG_TO_FILE_ONLY:
            log_dir = Path("logs") / timestamp / agent_args.env_id
            log_dir.mkdir(parents=True, exist_ok=True)
            sim_dir = Path("logs") / timestamp / "similarityheat"
            sim_dir.mkdir(parents=True, exist_ok=True)
            pram_drift_dir = Path("logs") / timestamp / "parameter_drift"
            pram_drift_dir.mkdir(parents=True, ok=True)
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
        else:
            log_dir = None
            sim_dir = None
            pram_drift_dir = None
            train_xlsx = None
            test_xlsx = None
            log_fh = None
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.StreamHandler(sys.stdout)
                ]
            )
            logging.info("Logging started. Outputting to terminal only.")

        # No communication module, just run the agent
        runner = Runner(agent_args, train_xlsx, test_xlsx, sim_dir, pram_drift_dir,None, None)
        if agent_args.only_render:
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
            if agent_args.env_id not in ll:
                runner.render(epoch=0, obj_set="train")
            runner.render(epoch=0, obj_set="test")
        else:
            if LOG_TO_FILE_ONLY:
                print("Training started. Log file: %s" % (log_dir / "log.txt"))
            else:
                print("Training started. Logging to terminal only.")
            runner.run()
        if LOG_TO_FILE_ONLY and log_fh is not None:
            log_fh.close()
        return

    # --- Multiprocess mode (original logic) ---
    cuda_devices = [d.strip() for d in args.cuda_devices.split(',')]
    hosts = [h.strip() for h in args.hosts.split(',')]
    if args.num_agents > len(ENVIRONMENTS):
        raise ValueError(f"Requested {args.num_agents} agents, but only {len(ENVIRONMENTS)} environments available.")
    if len(cuda_devices) < args.num_agents:
        print(f"Warning: {args.num_agents} agents but only {len(cuda_devices)} CUDA devices. Sharing GPUs.")
    if len(hosts) < args.num_agents:
        hosts = hosts * (args.num_agents // len(hosts) + 1)
    hosts = hosts[:args.num_agents]
    
    addrs = [(hosts[i], args.base_port + i) for i in range(args.num_agents)]
    
    procs = []
    ctx = mp.get_context('spawn')
    for i in range(args.num_agents):
        device = cuda_devices[i % len(cuda_devices)]
        agent_name = f"{args.base_name}_{i}"
        agent_seed = args.base_seed + i
        env_id = ENVIRONMENTS[i]
        addr_i = addrs[i]
        peer_addrs = [addrs[j] for j in range(args.num_agents) if j != i]
        agent_cli_args = [
            "--env-id", env_id,
            "--name", agent_name,
            "--seed", str(agent_seed),
            "--comm-interval", str(args.comm_interval),
            "--agent-id", str(i),
            "--all-envs", ",".join(ENVIRONMENTS[:args.num_agents]),
        ] + unknown
        print(f"Launching agent {i}: name={agent_name}, seed={agent_seed}, env_id={env_id}, CUDA_VISIBLE_DEVICES={device}")
        proc = mp.Process(target=worker, args=(agent_cli_args, timestamp, device, addr_i, peer_addrs, args.server_side_selection, args.sim_threshold))
        proc.start()
        procs.append(proc)

    # Real-time heatmap monitoring thread
    def realtime_heatmap_monitor(num_agents, sim_dir="logs/" + timestamp + "/similarityheat"):
        import time
        from pathlib import Path
        import re
        sim_dir = Path(sim_dir)
        print(sim_dir)
        seen_episodes = set()
        # Get env_ids for the current run
        env_ids = ENVIRONMENTS[:num_agents]
        while any(p.is_alive() for p in procs):
            emb_files = list(sim_dir.glob("embedding_agent_*_ep_*.npy"))
            # Find all episode numbers for which we have all agent embeddings
            ep_to_agents = {}
            for f in emb_files:
                m_ep = re.search(r"ep_(\d+)", str(f))
                m_id = re.search(r"agent_(\d+)", str(f))
                if m_ep and m_id:
                    ep = int(m_ep.group(1))
                    aid = int(m_id.group(1))
                    ep_to_agents.setdefault(ep, set()).add(aid)
            for ep, agent_ids in ep_to_agents.items():
                if len(agent_ids) == num_agents and ep not in seen_episodes:
                    if generate_similarity_heatmap(num_agents, ep, sim_dir=sim_dir, env_ids=env_ids):
                        seen_episodes.add(ep)
            time.sleep(2)  # Polling interval

    # --- Parameter Drift Monitoring Thread ---
    def realtime_parameter_drift_monitor(num_agents, drift_dir="logs/" + timestamp + "/parameter_drift"):
        import time
        from pathlib import Path
        import re
        import subprocess
        drift_dir = Path(drift_dir)
        print(drift_dir)
        seen_episodes = set()
        while any(p.is_alive() for p in procs):
            ckpt_files = list(drift_dir.glob("policy_agent_*_ep_*.safetensors"))
            # Find all episode numbers for which we have all agent checkpoints
            ep_to_agents = {}
            for f in ckpt_files:
                m_ep = re.search(r"ep_(\d+)", str(f))
                m_id = re.search(r"agent_(\d+)", str(f))
                if m_ep and m_id:
                    ep = int(m_ep.group(1))
                    aid = int(m_id.group(1))
                    ep_to_agents.setdefault(ep, set()).add(aid)
            for ep, agent_ids in ep_to_agents.items():
                if len(agent_ids) == num_agents and ep not in seen_episodes:
                    # All checkpoints for this episode are present
                    # Run the visualization script
                    ckpt_paths = [str((drift_dir / f"policy_agent_{aid}_ep_{ep}.safetensors").resolve()) for aid in sorted(agent_ids)]
                    out_dir = (drift_dir / f"ep_{ep}_drift").resolve()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        import os
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        visualize_path = os.path.join(script_dir, "visualize.py")
                        subprocess.run(
                            ["python", visualize_path, *ckpt_paths, "-o", str(out_dir)],
                            check=True,
                            cwd=script_dir  # Working directory for script, but all paths are now absolute
                        )
                        print(f"[ParameterDrift] Visualized drift for episode {ep}, output in {out_dir}")
                        seen_episodes.add(ep)
                    except Exception as e:
                        print(f"[ParameterDrift] Failed to visualize drift for episode {ep}: {e}")
            time.sleep(2)  # Polling interval
    
    monitor_thread = threading.Thread(target=realtime_heatmap_monitor, args=(args.num_agents, "logs/" + timestamp + "/similarityheat"), daemon=True)
    monitor_thread.start()

    # Start parameter drift monitor thread
    drift_monitor_thread = threading.Thread(target=realtime_parameter_drift_monitor, args=(args.num_agents, "logs/" + timestamp + "/parameter_drift"), daemon=True)
    drift_monitor_thread.start()

    for i, proc in enumerate(procs):
        proc.join()
        print(f"Agent {i} (PID {proc.pid}) exited with code {proc.exitcode}")
    
    

    monitor_thread.join(timeout=2)
    drift_monitor_thread.join(timeout=2)


if __name__ == "__main__":
    # Launch agents as before
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main(timestamp)
    
    # After all agents have finished, try to generate heatmaps for all episodes
    # (Optionally, this could be done in real-time, but here we do it after training)
    