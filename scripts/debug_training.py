import torch
import matplotlib.pyplot as plt
import os

def analyze_checkpoints(checkpoint_dir, train_type):
    """Analyze multiple checkpoints to understand training progression"""
    
    # Find all checkpoints
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    epochs = []
    losses = []
    
    for ckpt in ckpts:
        epoch = int(ckpt.split("_")[-1].split(".")[0])
        epochs.append(epoch)
        
        # You can extract more info from checkpoints if stored
        print(f"Checkpoint: {ckpt} (Epoch {epoch})")
    
    print(f"Available epochs: {epochs}")
    print(f"Training appears to have {len(epochs)} saved checkpoints")
    
    # Check wandb logs for actual loss curves
    print("\n🔍 Check your wandb dashboard for:")
    print("- epoch_loss trend")
    print("- unique_codes utilization")
    print("- learning_rate changes")
    print("- Signs of overfitting (train vs val divergence)")

if __name__ == "__main__":
    # Analyze your current training
    checkpoint_dir = "checkpoints/all_checkpoints_vqvae_moe_med"
    analyze_checkpoints(checkpoint_dir, "MOE_med")
