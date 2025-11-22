import multiprocessing
import time

def sender(q_emb, agent_id, episode):
    # Simulate preparing data
    vi_list = [1.0, 2.0, 3.0]
    performance = 99.0
    mask_id = f"{agent_id}_{episode}"
    serialized_lora = [[0.1, 0.2], [0.3, 0.4]]
    print(f"Sender: putting to q_emb: {vi_list}, {performance}, {mask_id}, {serialized_lora}")
    q_emb.put((vi_list, performance, mask_id, serialized_lora))

def receiver(q_emb, q_mask):
    # Simulate communication module
    try:
        item = q_emb.get(timeout=5)
        vi_list, performance, mask_id, serialized_lora = item
        print(f"Receiver: got from q_emb: {item}")
        # Simulate preparing a response for q_mask
        peer_agent_id = 42
        peer_perf = 88.0
        peer_mask_ser = [[0.5, 0.6], [0.7, 0.8]]
        q_mask.put((peer_agent_id, peer_perf, peer_mask_ser))
        print(f"Receiver: put to q_mask: {(peer_agent_id, peer_perf, peer_mask_ser)}")
    except Exception as e:
        print(f"Receiver: error: {e}")

def main():
    q_emb = multiprocessing.Queue()
    q_mask = multiprocessing.Queue()

    # Start sender
    sender_proc = multiprocessing.Process(target=sender, args=(q_emb, 0, 1))
    # Start receiver
    receiver_proc = multiprocessing.Process(target=receiver, args=(q_emb, q_mask))

    sender_proc.start()
    receiver_proc.start()

    sender_proc.join()
    receiver_proc.join()

    # Simulate agent waiting for peer mask
    try:
        item = q_mask.get(timeout=5)
        peer_agent_id, peer_perf, peer_mask_ser = item
        print(f"Main: got from q_mask: {item}")
    except Exception as e:
        print(f"Main: error: {e}")

if __name__ == "__main__":
    main()