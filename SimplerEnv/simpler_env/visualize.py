

#!/usr/bin/env python3
"""
visualize_safetensors_drift.py

Usage:
    python visualize_safetensors_drift.py ckpt1.safetensors ckpt2.safetensors ... -o out_dir

Produces plots and a CSV with top drifting parameters.
"""
import os
import sys
import argparse
from safetensors.torch import load_file
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["figure.max_open_warning"] = 50

import time

def wait_for_stable_file(path, wait_time=5, max_attempts=20):
    """Wait until the file size is stable for wait_time seconds (default 5s)."""
    from pathlib import Path
    last_size = -1
    stable_count = 0
    attempts = 0
    while attempts < max_attempts:
        if not Path(path).exists():
            time.sleep(1)
            attempts += 1
            continue
        size = Path(path).stat().st_size
        if size == last_size and size > 0:
            stable_count += 1
            if stable_count >= wait_time:
                return
        else:
            stable_count = 0
        last_size = size
        time.sleep(1)
        attempts += 1
    print(f"[WARNING] File {path} may not be fully written after waiting.")

def load_state(path):
    # returns dict: name->torch.Tensor on CPU
    import os
    from pathlib import Path
    wait_for_stable_file(path)
    if not Path(path).exists():
        abs_path = os.path.abspath(path)
        print(f"[ERROR] Checkpoint file not found: {path} (absolute: {abs_path})")
        raise FileNotFoundError(f"Checkpoint file not found: {path} (absolute: {abs_path})")
    sd = load_file(path)
    # ensure tensors are CPU and float32 where numeric:
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            sd[k] = v.detach().cpu().to(torch.float32)
        else:
            # safetensors sometimes returns numpy arrays; convert
            sd[k] = torch.as_tensor(v).cpu().to(torch.float32)
    return sd

def common_param_names(list_of_state_dicts):
    sets = [set(sd.keys()) for sd in list_of_state_dicts]
    common = set.intersection(*sets)
    return sorted(list(common))

def flatten_param_tensor(tensor):
    return tensor.view(-1).cpu().numpy()

def main(args):
    os.makedirs(args.out, exist_ok=True)
    paths = args.checkpoints
    names = [os.path.basename(p) for p in paths]
    print("Loading", len(paths), "checkpoints...")

    sds = [load_state(p) for p in paths]
    print("Loaded.")

    common = common_param_names(sds)
    if not common:
        raise RuntimeError("No common parameter names across checkpoints. Check your files.")
    print(f"{len(common)} parameters common across checkpoints.")

    # 1) compute norms and store per-param per-checkpoint
    param_norms = pd.DataFrame(index=common, columns=names, dtype=float)
    for i, sd in enumerate(sds):
        for pname in common:
            t = sd[pname]
            param_norms.iloc[:, i].loc[pname] = float(torch.norm(t).item())

    # Save raw norms
    param_norms.to_csv(os.path.join(args.out, "param_l2_norms.csv"))

    # 2) compute per-parameter drift measures
    # absolute changes between subsequent checkpoints
    deltas = param_norms.diff(axis=1).iloc[:, 1:]  # NaN in first column
    # relative changes (delta / previous norm)
    rel_changes = deltas.div(param_norms.iloc[:, :-1].values, axis=0)

    # total drift across the whole run (sum of absolute deltas)
    total_abs_drift = deltas.abs().sum(axis=1).sort_values(ascending=False)
    total_rel_drift = rel_changes.abs().sum(axis=1).sort_values(ascending=False)

    # 3) Top-k drifting params
    topk = args.topk
    topk_names = total_abs_drift.head(topk).index.tolist()
    pd.DataFrame({
        "total_abs_drift": total_abs_drift,
        "total_rel_drift": total_rel_drift
    }).to_csv(os.path.join(args.out, "param_total_drifts.csv"))

    # 4) Plotting: line plots for top-k parameters (norm vs checkpoint)
    def plot_lines(param_list, fname, title):
        plt.figure(figsize=(10, 6))
        for pname in param_list:
            plt.plot(names, param_norms.loc[pname].values, label=pname)
        plt.xlabel("checkpoint")
        plt.ylabel("L2 norm")
        plt.title(title)
        plt.legend(fontsize="small", ncol=1)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print("Wrote", fname)

    plot_lines(topk_names[:20], os.path.join(args.out, "topk_norm_lines.png"),
               f"Top {min(20, len(topk_names))} drifting params (L2 norm over checkpoints)")

    # 5) Heatmap of relative changes (params x checkpoints)
    # To keep heatmap readable, aggregate by module prefix (before first dot)
    module_prefix = [p.split('.')[0] for p in common]
    df = pd.DataFrame({"param": common, "module": module_prefix})
    agg = pd.concat([df.set_index("param"), rel_changes.abs()], axis=1, join="inner")
    # group by module and average absolute relative change across params per checkpoint
    module_change = agg.groupby("module").mean()
    plt.figure(figsize=(12, max(4, 0.3 * len(module_change))))
    plt.pcolormesh(range(module_change.shape[1]), range(module_change.shape[0]),
                   module_change.values, shading='auto')
    plt.yticks(np.arange(0.5, len(module_change)+0.5), module_change.index)
    plt.xticks(np.arange(0.5, len(module_change.columns)+0.5), module_change.columns, rotation=30)
    plt.colorbar(label="avg |relative change|")
    plt.title("Module-level average |relative change| between successive checkpoints")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "module_relative_change_heatmap.png"))
    plt.close()
    print("Wrote module_relative_change_heatmap.png")

    # 6) PCA on flattened full-parameter vector to visualize trajectory
    # Concatenate flattened params (in consistent order) into 1D vector per checkpoint.
    # For very large models we allow sampling a fraction of parameters.
    # sample_frac = float(args.sample_frac)
    # if sample_frac < 1.0:
    #     rng = np.random.RandomState(seed=0)
    #     # build index mask for elements of the full concatenated vector
    #     # easier approach: sample a subset of parameters (not elements) for performance
    #     sampled_params = sorted(list(common))[:max(1, int(len(common) * sample_frac))]
    #     concat_params = sampled_params
    #     print(f"Sampling {len(concat_params)} params out of {len(common)} (frac={sample_frac}) for PCA.")
    # else:
    #     concat_params = common

    # stacked = []
    # for sd in sds:
    #     parts = [flatten_param_tensor(sd[p]) for p in concat_params]
    #     stacked.append(np.concatenate(parts))
    # stacked = np.array(stacked)  # shape: (n_checkpoints, total_elems)
    # print("Stacked shape for PCA:", stacked.shape)

    # center data before PCA
    # pca = PCA(n_components=3)
    # pcs = pca.fit_transform(stacked)  # rows: checkpoints
    # # scatter 2D with lines
    # plt.figure(figsize=(8,6))
    # plt.plot(pcs[:,0], pcs[:,1], marker='o')
    # for i, lab in enumerate(names):
    #     plt.text(pcs[i,0], pcs[i,1], f"{i}:{lab}", fontsize=8)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("Parameter-space trajectory (PCA on flattened params)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.out, "pca_trajectory_pc1_pc2.png"))
    # plt.close()
    # print("Wrote pca_trajectory_pc1_pc2.png")

    # 7) Cosine similarity of each param vs checkpoint 0 (with sampling for speed)
    base_sd = sds[0]
    cos_sims = pd.DataFrame(index=common, columns=names, dtype=float)

    rng = np.random.default_rng(seed=42)
    sample_size = 100_000  # max elements sampled per tensor

    def sampled_flatten(tensor):
        arr = tensor.view(-1).cpu().numpy()
        if arr.size > sample_size:
            idx = rng.choice(arr.size, size=sample_size, replace=False)
            return arr[idx]
        return arr

    for i, sd in enumerate(sds):
        for pname in common:
            a = sampled_flatten(base_sd[pname])
            b = sampled_flatten(sd[pname])
            da = np.linalg.norm(a)
            db = np.linalg.norm(b)
            if da == 0 or db == 0:
                cos = np.nan
            else:
                cos = float(np.dot(a, b) / (da * db))
            cos_sims.iloc[:, i].loc[pname] = cos

    cos_sims.to_csv(os.path.join(args.out, "param_cosine_similarities_vs_init.csv"))

    # Example plot: histogram of cosine similarities at last checkpoint
    last = cos_sims.iloc[:, -1].dropna()
    plt.figure(figsize=(8,5))
    plt.hist(last.values, bins=50)
    plt.xlabel("Cosine similarity vs checkpoint 0")
    plt.title("Histogram of per-parameter cosine similarity (last checkpoint, sampled)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "cosine_similarity_hist_last.png"))
    plt.close()
    print("Wrote cosine_similarity_hist_last.png (sampled)")

    # 8) Save a CSV with top-k drifting param names and their norms across checkpoints
    topk_table = param_norms.loc[topk_names]
    topk_table["total_abs_drift"] = total_abs_drift.loc[topk_names]
    topk_table["total_rel_drift"] = total_rel_drift.loc[topk_names]
    topk_table.to_csv(os.path.join(args.out, f"top_{topk}_drifting_params.csv"))
    print("Wrote top_k drifting params CSV")

    print("Done. Outputs are in:", os.path.abspath(args.out))

    # Delete checkpoint files after visualization
    for ckpt_path in args.checkpoints:
        try:
            os.remove(ckpt_path)
            print(f"Deleted checkpoint file: {ckpt_path}")
        except Exception as e:
            print(f"Could not delete checkpoint file {ckpt_path}: {e}")



import matplotlib.pyplot as plt

def generate_similarity_heatmap(num_agents, episode, sim_dir="logs/similarityheat", env_ids=None):
    import numpy as np
    from pathlib import Path
    sim_dir = Path(sim_dir)
    print(sim_dir)
    embeddings = []
    for agent_id in range(num_agents):
        emb_path = sim_dir / f"embedding_agent_{agent_id}_ep_{episode}.npy"
        if not emb_path.exists():
            print(f"Embedding file missing: {emb_path}")
            return False  # Not all embeddings are present yet
        try:
            emb = np.load(emb_path)
        except Exception as e:
            print(f"Error loading embedding for agent {agent_id}, episode {episode}: {e}")
            return False
        if emb.ndim == 0 or emb.size == 0:
            print(f"Embedding for agent {agent_id}, episode {episode} is empty or invalid shape: {emb.shape}")
            return False
        if emb.ndim > 2:
            print(f"Embedding for agent {agent_id}, episode {episode} has too many dimensions: {emb.shape}")
            return False
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    print("Stacked embeddings shape:", embeddings.shape)
    # Compute cosine similarity matrix
    if embeddings.ndim != 2:
        print(f"Stacked embeddings are not 2D: {embeddings.shape}")
        return False
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(normed, normed.T)
    print("sim_matrix", sim_matrix)
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.title(f"Agent/Task Similarity Heatmap (Episode {episode})")
    plt.xlabel("Agent/Env")
    plt.ylabel("Agent/Env")
    if env_ids is not None and len(env_ids) == num_agents:
        plt.xticks(range(num_agents), env_ids, rotation=45, ha='right')
        plt.yticks(range(num_agents), env_ids)
    else:
        plt.xticks(range(num_agents), [f"A{i}" for i in range(num_agents)])
        plt.yticks(range(num_agents), [f"A{i}" for i in range(num_agents)])
    plt.tight_layout()
    out_path = sim_dir / f"heatmap_ep_{episode}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved similarity heatmap: {out_path}")
    # Delete embedding .npy files for this episode to save space
    for agent_id in range(num_agents):
        emb_path = sim_dir / f"embedding_agent_{agent_id}_ep_{episode}.npy"
        try:
            emb_path.unlink()
            print(f"Deleted embedding file: {emb_path}")
        except Exception as e:
            print(f"Could not delete {emb_path}: {e}")
    return True




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize parameter drift across safetensors checkpoints.")
    parser.add_argument("checkpoints", nargs="+", help="Paths to .safetensors checkpoints (in chronological order).")
    parser.add_argument("-o", "--out", default="drift_plots", help="Output directory.")
    parser.add_argument("--topk", type=int, default=200, help="How many top drifting params to include in CSV and line plots.")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of parameters to sample for PCA (1.0 = all). Useful for large models.")
    args = parser.parse_args()
    main(args)
