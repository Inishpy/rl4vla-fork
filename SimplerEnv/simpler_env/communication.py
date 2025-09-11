

import logging
import torch
import numpy as np
import time

def communicate(self, episode, current_success, timeout=60.0):
    try:
        # --- Skip communication for first few episodes (warmup) ---
        comm_start = getattr(self.args, "comm_start", 0)
        if episode < comm_start:
            logging.info(f"[share_and_receive] Skipping communication: episode {episode} < comm_start {comm_start}")
            return

        # Communication interval check
        if episode % self.args.comm_interval != 0:
            logging.info("[share_and_receive] Not a communication interval; returning early")
            return
        logging.info(f"[share_and_receive] Communication interval hit (comm_interval={self.args.comm_interval})")

        # If no data in FIFO, skip (mirrors your previous behavior)
        if len(self.buffer_fifo) == 0:
            logging.warning(f"[share_and_receive] Skipping: FIFO buffer has no data yet. size={len(self.buffer_fifo)}")
            return

        # --- Compute performance as mean success (keeps your previous logic) ---
        rewards = []
        N = 100
        try:
            if len(self.buffer_fifo) >= N:
                rewards = [self.buffer_fifo.rewards[(self.buffer_fifo.ptr - i - 1) % self.buffer_fifo.capacity][0] for i in range(N)]
            else:
                rewards = [self.buffer_fifo.rewards[i][0] for i in range(len(self.buffer_fifo))]
        except Exception:
            try:
                if len(self.buffer_fifo) >= N:
                    batch = self.buffer_fifo.sample(N)
                else:
                    batch = self.buffer_fifo.sample(len(self.buffer_fifo))
                rewards = batch["rewards"].flatten()
            except Exception as e:
                logging.debug(f"[share_and_receive] Couldn't extract rewards from buffer: {e}")
                rewards = []

        # performance based on current_success as before
        try:
            self.performance = current_success.float().mean().item() * 100.0
        except Exception:
            # fallback if current_success not tensor-like
            try:
                self.performance = float(current_success) * 100.0
            except Exception:
                self.performance = 0.0
        logging.info(f"[share_and_receive] Computed performance: {self.performance}")

    except Exception as e:
        logging.error(f"[share_and_receive] Exception during local computation: {e}", exc_info=True)
        raise

    # --- Prepare and send TEQ via q_emb (embedding, perf, mask_id, serialized_mask) ---
    try:
        # Ensure embedding exists
        if self.task_embedding is None:
            self.task_embedding = self.compute_task_embedding()

        # convert embedding to serializable list (float32)
        vi_list = self.task_embedding.cpu().numpy().astype(np.float32).tolist()

        # Prepare serialized mask (so comm server can send it on request)
        sparse_lora = self.make_sparse_lora()  # list of torch tensors
        serialized_lora = [p.cpu().detach().to(torch.float32).numpy().astype(np.float32) for p in sparse_lora]

        mask_id = f"{self.args.agent_id}_{episode}"
        # store locally for later traceability (comm module also receives it via Q_emb)
        self.current_mask_id = mask_id
        self.current_mask_ser = serialized_lora

        # Put TEQ into communication queue. communication_module should read this and
        # publish own_data['mask'] and other fields so the server can respond to MR/MTR.
        try:
            self.q_emb.put((vi_list, float(self.performance), mask_id, serialized_lora))
            logging.info(f"[share_and_receive] Put TEQ to q_emb (agent={self.args.agent_id}, episode={episode}, mask_id={mask_id})")
        except Exception as e:
            logging.error(f"[share_and_receive] Failed to put TEQ on q_emb: {e}", exc_info=True)
            # proceed; maybe comm module not running

    except Exception as e:
        logging.error(f"[share_and_receive] Exception preparing TEQ/mask: {e}", exc_info=True)
        raise

    # --- Optional: force_sharing_test injects fake peers into q_mask for testing ---
    if getattr(self.args, "force_sharing_test", False):
        try:
            # create two fake peers (structure: (peer_id, peer_perf, peer_mask_ser))
            test_peer_id = (self.args.agent_id + 1) % getattr(self.args, "num_agents", (self.args.agent_id + 2))
            noise = torch.from_numpy(np.random.randn(*self.task_embedding.shape).astype(np.float32)).to(self.task_embedding.device) * 0.01
            fake_mask = [ (p.cpu().numpy().astype(np.float32) * 0.0 + 0.001) for p in sparse_lora ]  # tiny fake mask arrays
            fake_perf = max(self.performance - 0.5, 0.1)
            try:
                self.q_mask.put((test_peer_id, float(fake_perf), fake_mask))
                logging.info(f"[share_and_receive] Injected fake peer {test_peer_id} into q_mask (perf={fake_perf})")
            except Exception as e:
                logging.warning(f"[share_and_receive] Failed to inject fake peer into q_mask: {e}")
            # second fake peer with slightly higher perf
            test_peer_id2 = (self.args.agent_id + 2) % getattr(self.args, "num_agents", (self.args.agent_id + 3))
            fake_perf2 = self.performance + 0.5
            try:
                self.q_mask.put((test_peer_id2, float(fake_perf2), fake_mask))
                logging.info(f"[share_and_receive] Injected fake peer {test_peer_id2} into q_mask (perf={fake_perf2})")
            except Exception as e:
                logging.warning(f"[share_and_receive] Failed to inject second fake peer into q_mask: {e}")
        except Exception as e:
            logging.warning(f"[share_and_receive] Error during force_sharing_test injection: {e}", exc_info=True)

    # --- Wait for incoming masks from q_mask (produced by communication_module) ---
    logging.info(f"[share_and_receive] Waiting up to {timeout}s for masks on q_mask")
    from queue import Empty
    start = time.time()
    received_items = []
    while time.time() - start < timeout:
        try:
            item = self.q_mask.get_nowait()
            # expected item: (peer_agent_id, peer_perf, peer_mask_ser)
            if item is None:
                continue
            try:
                peer_id_j, peer_perf_j, peer_mask_ser = item
            except Exception:
                logging.warning(f"[share_and_receive] Unexpected q_mask item format: {item}")
                continue
            logging.info(f"[share_and_receive] Received mask from peer {peer_id_j} perf={peer_perf_j}")
            received_items.append((int(peer_id_j), float(peer_perf_j), peer_mask_ser))
        except Empty:
            # no immediate item; sleep briefly
            time.sleep(0.05)
        except Exception as e:
            logging.error(f"[share_and_receive] Error reading q_mask: {e}", exc_info=True)
            time.sleep(0.1)

    logging.info(f"[share_and_receive] Collected {len(received_items)} masks from peers")

    # --- Accept and store received masks ---
    self.received_masks = {}   # mapping peer_id -> deserialized mask (list of numpy arrays)
    peer_id_to_perf = {}
    receiver_id = getattr(self.args, "agent_id", None)
    received_from_ids = []
    received_from_strs = []
    for peer_id, peer_perf, peer_mask_ser in received_items:
        try:
            # peer_mask_ser is already a list of numpy arrays (float32) from comm module
            peer_lora = [np.array(p, dtype=np.float32) for p in peer_mask_ser]
            self.received_masks[peer_id] = peer_lora
            peer_id_to_perf[peer_id] = float(peer_perf)
            received_from_ids.append(peer_id)
            received_from_strs.append(f"agent {peer_id} (perf={peer_perf:.3f})")
            logging.info(f"[share_and_receive] Agent {receiver_id} accepted mask from agent {peer_id} (perf={peer_perf:.3f})")
        except Exception as e:
            logging.warning(f"[share_and_receive] Failed to deserialize mask from {peer_id}: {e}", exc_info=True)

    # --- Set beta=0.01 for any new peer, then normalize so sum=1 ---
    for peer_id in received_from_ids:
        # Extend beta_weights if needed
        if peer_id >= len(self.beta_weights):
            self.beta_weights += [0.0] * (peer_id - len(self.beta_weights) + 1)
        # If this peer is new (beta was 0), set to 0.01
        if self.beta_weights[peer_id] == 0.0:
            self.beta_weights[peer_id] = 0.01
    # Normalize so sum is 1 (including own and all peers)
    total = sum(self.beta_weights)
    if total > 0:
        self.beta_weights = [b / total for b in self.beta_weights]
    if received_from_ids:
        logging.info(f"[share_and_receive] Agent {receiver_id} received masks from agents: {received_from_ids} before composing.")
    else:
        logging.info(f"[share_and_receive] Agent {receiver_id} did not receive any masks from peers before composing.")

    # --- Write to train.xlsx and test.xlsx before composing ---
    import pandas as pd
    excel_row = {
        "episode": episode,
        "received_from": ", ".join(received_from_strs) if received_from_strs else "no masks received"
    }
    for xlsx_path in [getattr(self, "train_xlsx", None), getattr(self, "test_xlsx", None)]:
        if xlsx_path is not None:
            try:
                try:
                    df = pd.read_excel(xlsx_path)
                except Exception:
                    df = pd.DataFrame(columns=["episode", "received_from"])
                df = pd.concat([df, pd.DataFrame([excel_row])], ignore_index=True)
                df.to_excel(xlsx_path, index=False)
                logging.info(f"[share_and_receive] Appended mask receipt info to {xlsx_path}: {excel_row}")
            except Exception as e:
                logging.warning(f"[share_and_receive] Failed to append mask receipt info to {xlsx_path}: {e}", exc_info=True)

    # --- Update beta weights using own perf + accepted peers ---
    try:
        total_perf = float(self.performance) + sum(peer_id_to_perf.values())
        logging.info(f"[share_and_receive] Calculated total_perf = {total_perf:.6f}")

        if total_perf > 0:
            # Reset weights (keep same length)
            if isinstance(self.beta_weights, np.ndarray):
                self.beta_weights.fill(0.0)
            else:
                self.beta_weights = [0.0] * len(self.beta_weights)
            # set own weight keyed by task_idx (as before)
            self.beta_weights[self.task_idx] = float(self.performance) / (total_perf + 1e-12)
            for peer_id in self.received_masks:
                # peer_id should map to an index in beta_weights; original used peer_id directly
                if peer_id < len(self.beta_weights):
                    self.beta_weights[peer_id] = peer_id_to_perf.get(peer_id, 0.0) / (total_perf + 1e-12)
                else:
                    logging.debug(f"[share_and_receive] Peer id {peer_id} outside beta_weights length ({len(self.beta_weights)})")
        else:
            logging.warning("[share_and_receive] total_perf <= 0; leaving beta_weights unchanged.")
        logging.info(f"[share_and_receive] Beta weights updated: {self.beta_weights}")
    except Exception as e:
        logging.error(f"[share_and_receive] Error updating beta_weights: {e}", exc_info=True)

    # --- Compose policy using new weights and masks ---
    try:
        logging.info("[share_and_receive] Starting policy composition")
        # self.received_masks contains peer masks as deserialized numpy arrays
        # ensure compose_policy consumes self.received_masks correctly
        # Log detailed mapping of received masks before composing
        if self.received_masks:
            for sender_id in self.received_masks:
                logging.info(f"[share_and_receive] Agent {receiver_id} will use mask from agent {sender_id} in composition.")
        else:
            logging.info(f"[share_and_receive] Agent {receiver_id} has no peer masks to use in composition.")
        self.compose_policy()
        logging.info("[share_and_receive] Finished composition")
    except Exception as e:
        logging.error(f"[share_and_receive] Error during compose_policy: {e}", exc_info=True)

    # done
    return