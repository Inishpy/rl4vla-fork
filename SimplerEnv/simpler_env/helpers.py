
import numpy as np
from sklearn.decomposition import PCA
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
import math
# ----------------------------
# 1) Resize + Flatten
# ----------------------------
def obs_to_flat_resized(obs, target_size=(64, 64)):
    """
    Resize images to target_size and flatten
    obs: list or array of images (H, W, C)
    returns: np.array of shape (N, target_size[0]*target_size[1]*C)
    """
    states = np.array([
        cv2.resize(o, target_size).flatten() for o in obs
    ], dtype=np.float32) / 255.0
    return states


# ----------------------------
# 2) Flatten + PCA
# ----------------------------
def obs_to_flat_pca(obs, n_components=512):
    """
    Flatten images and reduce dimensionality with PCA
    obs: list or array of images (H, W, C)
    returns: np.array of shape (N, n_components)
    """
    flat_states = np.array([o.flatten() for o in obs], dtype=np.float32) / 255.0
    pca = PCA(n_components=n_components)
    states_reduced = pca.fit_transform(flat_states)
    return states_reduced


# ----------------------------
# 3) CNN Embeddings (ResNet18)
# ----------------------------
def obs_to_cnn_features(obs, device='cuda', batch_size=12):
    """
    obs: numpy array (N, H, W, C), values 0–255
    returns: np.array (N, 512) ResNet18 features
    """
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove classifier
    model.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Convert (N,H,W,C) → (N,C,H,W)
    imgs = torch.tensor(obs, dtype=torch.uint8).permute(0, 3, 1, 2)

    features = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size].to(device)
            batch = transform(batch)                # resize + normalize
            feats = model(batch).squeeze()          # (batch, 512, 1, 1)
            feats = feats.view(feats.size(0), -1)   # (batch, 512)
            features.append(feats.cpu().numpy())

    return np.vstack(features).astype(np.float32)
def to_tensor_on_device(arr_or_tensor, device):
    """
    Convert a numpy array or torch tensor to a torch.float32 tensor on `device`.
    Ensures contiguity and detaches grad if needed.
    """
    if isinstance(arr_or_tensor, torch.Tensor):
        t = arr_or_tensor.detach()
    else:
        # numpy -> tensor (works for np.ndarray)
        t = torch.as_tensor(arr_or_tensor)
    return t.contiguous().to(device=device, dtype=torch.float32)


def safe_compute_f(ref_size, gamma, X, ref, device=None, chunk_cols=1024):
    """
    Compute f = ( (ref_size * gamma).T @ X - ref ) / sqrt(ref_size)
    in a robust, chunked way.

    Parameters:
      - ref_size: scalar (int/float/torch scalar)
      - gamma: torch tensor or numpy array
      - X: torch tensor or numpy array (2D)
      - ref: torch tensor or numpy array (2D) — must match output shape for subtraction
      - device: torch.device or None (auto-selects CUDA if available)
      - chunk_cols: number of columns of X to process per chunk (tune for memory)
    Returns:
      - f: torch.Tensor on `device`
    """
    # auto device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert to tensors
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.as_tensor(gamma)
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X)
    if not isinstance(ref, torch.Tensor):
        ref = torch.as_tensor(ref)

    # float32 and contiguous
    gamma = gamma.contiguous().to(dtype=torch.float32)
    X = X.contiguous().to(dtype=torch.float32)
    ref = ref.contiguous().to(dtype=torch.float32)

    # move to device
    gamma = gamma.to(device)
    X = X.to(device)
    ref = ref.to(device)

    # scalar ref_size -> float tensor on same device
    ref_size_f = float(ref_size) if not isinstance(ref_size, torch.Tensor) else float(ref_size.item())
    denom = torch.sqrt(torch.tensor(ref_size_f, dtype=torch.float32, device=device))

    # Build A = (ref_size * gamma).T
    A = (gamma * ref_size_f).T.contiguous()   # A shape: (p, q) depending on gamma

    # Determine orientation for matmul: we want A @ X_part
    # Ensure A.shape[1] == X.shape[0]; otherwise try transposing X or A appropriately
    if A.shape[1] != X.shape[0]:
        # maybe X is (N_samples, features) and A expects features x something.
        # If needed, transpose X to match. This depends on your intended equation.
        # Here, we'll try X = X.T if that aligns dims.
        if A.shape[1] == X.T.shape[0]:
            X_proc = X.T
        else:
            raise RuntimeError(f"Shape mismatch: A.shape={tuple(A.shape)}, X.shape={tuple(X.shape)}; cannot align for matmul.")
    else:
        X_proc = X

    # chunk along columns of X_proc (i.e., last dim)
    total_cols = X_proc.shape[1]
    parts = []
    for i in range(0, total_cols, chunk_cols):
        part = X_proc[:, i:i+chunk_cols]          # shape (A.shape[1], chunk_size)
        # compute A @ part -> shape (A.shape[0], chunk_size)
        parts.append(torch.matmul(A, part))
    out = torch.cat(parts, dim=1)                # shape (A.shape[0], total_cols)

    # Align ref for subtraction. ref must have same shape as out:
    if tuple(ref.shape) != tuple(out.shape):
        # try transposing ref if it matches
        if tuple(ref.T.shape) == tuple(out.shape):
            ref_aligned = ref.T
        else:
            raise RuntimeError(f"ref shape {tuple(ref.shape)} does not match matmul output shape {tuple(out.shape)}; align them first.")
    else:
        ref_aligned = ref

    f = (out - ref_aligned) / denom
    return f



import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Try imports; if missing, user will need to pip install them.
try:
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    CLIPProcessor = None
    CLIPModel = None

try:
    import timm
except Exception:
    timm = None

# -------------------------
# Helper: ensure device, batchify
# -------------------------
def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)

def _batchify(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# -------------------------
# 1) CLIP-based encoder
# -------------------------
def obs_to_clip_features(obs,
                         model_name='openai/clip-vit-base-patch32',
                         device=None,
                         batch_size=16,
                         normalize=True):
    """
    Encode images using a CLIP image encoder via HuggingFace transformers.
    - obs: iterable of images as numpy arrays (H,W,C) uint8 or floats 0..255, or PIL images.
    - model_name: HF model id
    - device: 'cuda' or 'cpu'
    - batch_size: process in batches
    - normalize: L2-normalize embeddings (CLIP style)
    Returns: np.array shape (N, embedding_dim), dtype float32
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if CLIPModel is None or CLIPProcessor is None:
        raise ImportError("transformers not found. pip install transformers")

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    features = []
    for batch in _batchify(list(obs), batch_size):
        # Convert to PIL if needed (CLIPProcessor expects PIL or array)
        imgs = [Image.fromarray(b) if isinstance(b, np.ndarray) else b for b in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)  # (B, D)
            if normalize:
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            features.append(image_embeds.cpu().float().numpy())
    return np.vstack(features).astype(np.float32)


# -------------------------
# 2) ViT encoder via timm (generic), with global pooling
# # -------------------------
def obs_to_vit_features(obs,
                        model_name='vit_base_patch16_224',
                        device=None,
                        batch_size=16,
                        pool='cls',
                        pretrained=True):
    """
    Encode images with a vision transformer from timm.
    - obs: iterable of images as numpy arrays (H,W,C) uint8 or floats 0..255, or PIL images.
    - model_name: timm model name (e.g. 'vit_base_patch16_224', 'swin_base_patch4_window7_224', etc.)
    - pool: 'cls' (use cls token), 'mean' (mean pool tokens), or 'gap' (global avg pool if model supports)
    Returns: np.array (N, D)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if timm is None:
        raise ImportError("timm not found. pip install timm")

    # Build model
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')  # keep tokens
    model.eval().to(device)
    
    # Basic preprocessing (resize to model default input size if available)
    # We'll assume model default is 224 if not known; user can adjust.
    input_size = model.default_cfg.get('input_size', (3, 224, 224)) if hasattr(model, 'default_cfg') else (3, 224, 224)
    size = (input_size[1], input_size[2])

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=model.default_cfg.get('mean', (0.485,0.456,0.406)),
                             std=model.default_cfg.get('std', (0.229,0.224,0.225)))
    ])

    features = []
    for batch in _batchify(list(obs), batch_size):
        imgs = torch.stack([preprocess(b) if isinstance(b, np.ndarray) else preprocess(np.array(b)) for b in batch]).to(device)
        with torch.no_grad():
            # many timm models return (B, C, H, W) features if num_classes=0 and global_pool=''
            out = model.forward_features(imgs)  # shape depends on architecture
            # out for ViT-like often (B, tokens, dim) or (B, dim, h, w) for conv-based models
            if out.ndim == 3:  # (B, tokens, dim) -> choose pool
                if pool == 'cls':
                    img_emb = out[:, 0, :]  # cls token
                elif pool == 'mean':
                    img_emb = out.mean(dim=1)
                else:
                    img_emb = out.mean(dim=1)
            elif out.ndim == 4:  # (B, dim, h, w) -> global pool
                img_emb = out.mean(dim=[2,3])
            else:
                # fallback: flatten last dims and global mean
                img_emb = out.reshape(out.shape[0], out.shape[1], -1).mean(dim=-1)

            features.append(img_emb.cpu().float().numpy())
    return np.vstack(features).astype(np.float32)


# -------------------------
# 3) Small projection head for mapping image embeddings -> target_dim
# -------------------------
class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, norm=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = norm
        if norm:
            self.ln = nn.LayerNorm(out_dim)
    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.ln(x)
        return x

def project_features(features_np, target_dim=1024, device=None, use_norm=True):
    """
    Project numpy features (N, D) to (N, target_dim) using a linear layer.
    Returns projected numpy array (N, target_dim) float32.
    Note: weights are randomly initialized. For best results, train/tune or load a pretrained projection.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = torch.from_numpy(features_np).float().to(device)
    projector = SimpleProjector(features.shape[1], target_dim, norm=use_norm).to(device)
    projector.eval()
    with torch.no_grad():
        out = projector(features).cpu().numpy()
    return out.astype(np.float32)



#4) DinoV3 ===========================

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
