import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora.layer import LoraLayer

class MultiHeadLoraLinear(LoraLayer):
    """
    A Linear layer supporting a variable number of LoRA adapters, with per-layer trainable betas for blending.
    Each adapter is identified by a string key. The base layer is always used, and each adapter can be blended in.
    """
    def __init__(self, base_layer: nn.Module, adapter_names, r_list, lora_alpha_list, lora_dropout_list=None, init_lora_weights_list=None):
        """
        base_layer: nn.Linear instance
        adapter_names: list of str, names for each LoRA adapter
        r_list: list of int, rank for each adapter
        lora_alpha_list: list of int, alpha for each adapter
        lora_dropout_list: list of float, dropout for each adapter (optional, default 0.0)
        init_lora_weights_list: list of bool/str, initialization for each adapter (optional, default True)
        """
        super().__init__(base_layer)
        self.adapter_names = adapter_names
        self.num_adapters = len(adapter_names)
        if lora_dropout_list is None:
            lora_dropout_list = [0.0] * self.num_adapters
        if init_lora_weights_list is None:
            init_lora_weights_list = [True] * self.num_adapters

        # Setup each adapter
        for i, name in enumerate(adapter_names):
            self.update_layer(
                name,
                r_list[i],
                lora_alpha_list[i],
                lora_dropout_list[i],
                init_lora_weights_list[i],
                use_rslora=False,
                use_dora=False,
            )

        # Per-layer trainable betas (including base)
        # beta_base for base, then one for each adapter
        self.beta_base = nn.Parameter(torch.tensor(1.0 / (self.num_adapters + 1), dtype=torch.float32))
        self.beta = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 / (self.num_adapters + 1), dtype=torch.float32))
            for _ in range(self.num_adapters)
        ])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Compute base output
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        # Weighted sum: base + sum_i beta_i * LoRA_i
        b_base = torch.clamp(self.beta_base, 0.0, 1.0)
        result = b_base.to(result.dtype) * result
        for i, name in enumerate(self.adapter_names):
            if name not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[name]
            lora_B = self.lora_B[name]
            dropout = self.lora_dropout[name]
            scaling = self.scaling[name]
            x_lora = x.to(lora_A.weight.dtype)
            b = torch.clamp(self.beta[i], 0.0, 1.0)
            lora_out = lora_B(lora_A(dropout(x_lora))) * scaling
            result = result + b.to(result.dtype) * lora_out.to(result.dtype)
        return result

    def get_betas(self):
        """Return the current beta weights as a list (base, [adapters...])"""
        return [self.beta_base.item()] + [b.item() for b in self.beta]
