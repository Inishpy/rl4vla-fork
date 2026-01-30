import torch
import torch.nn as nn
from SimplerEnv.simpler_env.policies.openvla.multihead_lora import MultiHeadLoraLinear

def test_multihead_lora_linear():
    # Create a base linear layer
    base = nn.Linear(8, 4, bias=False)
    # 2 adapters: "a1" and "a2"
    adapter_names = ["a1", "a2"]
    r_list = [2, 2]
    lora_alpha_list = [4, 4]
    lora_dropout_list = [0.0, 0.0]
    # Create the multihead lora layer
    mhl = MultiHeadLoraLinear(
        base,
        adapter_names=adapter_names,
        r_list=r_list,
        lora_alpha_list=lora_alpha_list,
        lora_dropout_list=lora_dropout_list,
    )
    # Print initial betas
    print("Initial betas:", mhl.get_betas())
    # Forward pass
    x = torch.randn(3, 8)
    y = mhl(x)
    print("Output shape:", y.shape)
    # Check gradients
    y.sum().backward()
    print("Gradients computed for base weight:", mhl.base_layer.weight.grad is not None)
    for i, name in enumerate(adapter_names):
        print(f"Gradients for lora_A {name}:", mhl.lora_A[name].weight.grad is not None)
        print(f"Gradients for lora_B {name}:", mhl.lora_B[name].weight.grad is not None)
    print("Gradients for betas:", [b.grad for b in [mhl.beta_base]+list(mhl.beta)])

if __name__ == "__main__":
    test_multihead_lora_linear()
