## MultiHeadLoraLinear Usage and Migration Notes

### What is `MultiHeadLoraLinear`?
`MultiHeadLoraLinear` is a drop-in replacement for a standard LoRA-wrapped `nn.Linear` layer, supporting a variable number of LoRA adapters per layer, with per-layer trainable blending weights (betas) for the base and each adapter.

### How to use

1. **Instantiate your base layer:**
   ```python
   import torch.nn as nn
   base = nn.Linear(in_features, out_features, bias=False)
   ```

2. **Define LoRA adapter configs:**
   ```python
   adapter_names = ["lora1", "lora2", ...]  # any number of adapters
   r_list = [rank1, rank2, ...]               # LoRA rank for each
   lora_alpha_list = [alpha1, alpha2, ...]    # LoRA alpha for each
   lora_dropout_list = [dropout1, dropout2, ...]  # (optional)
   init_lora_weights_list = [True, ...]           # (optional)
   ```

3. **Create the MultiHeadLoraLinear:**
   ```python
   from SimplerEnv.simpler_env.policies.openvla.multihead_lora import MultiHeadLoraLinear
   mhl = MultiHeadLoraLinear(
       base,
       adapter_names=adapter_names,
       r_list=r_list,
       lora_alpha_list=lora_alpha_list,
       lora_dropout_list=lora_dropout_list,
       init_lora_weights_list=init_lora_weights_list,
   )
   ```

4. **Forward pass:**
   ```python
   output = mhl(input_tensor)
   ```

5. **Trainable betas:**
   - Each layer has a trainable `beta_base` (for the base) and one `beta` per adapter.
   - These are optimized during training.
   - You can inspect them with `mhl.get_betas()`.

### Migration from standard LoRA

- Replace any instance of a LoRA-wrapped `nn.Linear` with `MultiHeadLoraLinear`.
- Instead of calling `update_layer` for each adapter, pass all adapter configs at construction.
- The forward pass and parameter optimization remain unchanged.

### Notes
- This module currently supports only `nn.Linear` layers. For Conv2d or Embedding, similar wrappers can be implemented.
- All LoRA adapters and the base are blended in a single forward pass, avoiding redundant computation.
- Per-layer betas allow flexible, learnable composition of multiple LoRA adapters.
