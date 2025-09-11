


# dinov3_worker.py
import sys
import json
import numpy as np
from pathlib import Path

# optional: lazy import torch here if needed
import torch
import torch.nn.functional as F
from PIL import Image
# Import your helper functions (_batchify, _to_pil, etc.) or include them here


def _batchify(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def _to_pil(img):
    # Accept np.ndarray (H,W,C uint8 / float 0..255) or PIL.Image
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            return Image.fromarray(img)
        else:
            # assume floats in 0..255
            arr = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
    raise ValueError("Unsupported image type: %s" % type(img))



def obs_to_dinov3_features(obs,
                           model_name='facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
                           device=None,
                           batch_size=16,
                           normalize=True,
                           use_pipeline=True):
    """
    Encode images using a DINOv3 image encoder via HuggingFace transformers.
    - obs: iterable of images as numpy arrays (H,W,C) uint8 or floats 0..255, or PIL images.
    - model_name: HF model id (examples: 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
                  'facebook/dinov3-vits16-pretrain-lvd1689m', 'facebook/dinov3-vitb16-pretrain-lvd1689m', ...)
    - device: 'cuda' or 'cpu' (default auto)
    - batch_size: process in batches
    - normalize: L2-normalize embeddings
    - use_pipeline: try HF pipeline('image-feature-extraction') first (simpler), else AutoImageProcessor+AutoModel
    Returns: np.array shape (N, embedding_dim), dtype float32
    """
    # choose device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # lazy imports with helpful errors
    try:
        from transformers import pipeline, AutoImageProcessor, AutoModel
        from transformers.image_utils import load_image  # not used but indicates availability
    except Exception as e:
        raise ImportError("transformers not installed or import failed. "
                          "Install recent transformers (pip install transformers) or "
                          "pip install --upgrade git+https://github.com/huggingface/transformers.git") from e

    features = []
    imgs = list(obs)

    # First try the simple pipeline API (it handles processor+model + batching)
    if use_pipeline:
        try:
            # dtype=torch.bfloat16 is sometimes recommended for big models, but we default to float32 here
            pipe = pipeline(task="image-feature-extraction",
                            model=model_name,
                            device=0 if device.startswith('cuda') else -1)
            for batch in _batchify(imgs, batch_size):
                pil_batch = [_to_pil(b) for b in batch]
                # pipeline returns list of embeddings per image (or tensors depending on HF version)
                out = pipe(pil_batch)  # list of arrays or dicts depending on transformers version
                # Normalize and stack robustly
                batch_feats = []
                for item in out:
                    # pipeline may return a list/ndarray directly or a dict with "last_hidden_state"/"pooled_output"
                    if isinstance(item, dict):
                        # try common keys
                        if 'pooled_output' in item:
                            vec = np.asarray(item['pooled_output'])
                        elif 'last_hidden_state' in item:
                            # mean-pool tokens
                            arr = np.asarray(item['last_hidden_state'])
                            vec = arr.mean(axis=0)
                        elif 'embed' in item:
                            vec = np.asarray(item['embed'])
                        else:
                            # fallback: try to flatten the dict to an array
                            # (unlikely - pipeline usually returns arrays)
                            vec = np.asarray(item)
                    else:
                        vec = np.asarray(item)
                        # If pipeline returned tokens (H, W, D) try global mean
                        if vec.ndim > 1:
                            vec = vec.mean(axis=tuple(range(vec.ndim - 1)))
                    batch_feats.append(vec.astype(np.float32))
                batch_feats = np.vstack(batch_feats)  # (B, D)
                if normalize:
                    norms = np.linalg.norm(batch_feats, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    batch_feats = batch_feats / norms
                features.append(batch_feats)
            return np.vstack(features).astype(np.float32)
        except Exception as e:
            # pipeline may fail if HF transform support is older or weights gated — fall back to AutoModel approach
            # print a warning and continue to fallback path
            # (do not raise; attempt fallback)
            # You can uncomment next line to see the pipeline error during debugging:
            # print("Pipeline failed, falling back to AutoModel path:", e)
            pass

    # Fallback: manually load processor + model and do forward passes
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    for batch in _batchify(imgs, batch_size):
        pil_batch = [_to_pil(b) for b in batch]
        inputs = processor(images=pil_batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Many HF vision models expose either `pooler_output` or `last_hidden_state`.
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output  # (B, D)
            else:
                # fallback: mean pooling over patch tokens (ignore any class token if present)
                last = outputs.last_hidden_state  # (B, seq_len, D)
                # If model has a class token at position 0 (common), you may prefer last[:,0,:]
                # but DINOv3 models typically produce strong patch features: we do mean pooling by default.
                pooled = last.mean(dim=1)  # (B, D)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            features.append(pooled.cpu().float().numpy())

    return np.vstack(features).astype(np.float32)


def main():
    # Read input from command line argument (JSON-encoded list of images)
    input_path = Path(sys.argv[1])
    obs = np.load(input_path, allow_pickle=True)  # e.g., saved numpy array of images

    features = obs_to_dinov3_features(obs, model_name="facebook/dinov3-vitl16-pretrain-sat493m")

    # Save output to a temporary file (or stdout)
    output_path = Path(sys.argv[2])
    np.save(output_path, features)

if __name__ == "__main__":
    main()