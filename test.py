import wandb
wandb.login()

import wandb
import random
import time

# Initialize W&B run
wandb.init(
    project="connection-test",   # change this if you want
    name="wandb-connection-check",
    config={
        "purpose": "Test W&B connection",
        "attempt": 1
    }
)

# Simulate logging metrics
for step in range(5):
    wandb.log({
        "step": step,
        "loss": random.random(),
        "accuracy": random.random()
    })
    print(f"Logged step {step}")
    time.sleep(1)

# Finish the run
wandb.finish()

print("\n✅ Test complete! Check your W&B dashboard to confirm the run appeared:")
print("   https://wandb.ai/home")
