import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <script_name>")
        print("Available scripts: train_vqvae, train_vqvae_cnn")
        sys.exit(1)

    script_name = sys.argv[1]

    # Map script names to file paths
    scripts = {
        "train_vqvae": "scripts/train_vqvae.py",
        "train_vqvae_cnn": "scripts/train_vqvae_cnn.py",
    }

    if script_name not in scripts:
        print(f"Error: Unknown script '{script_name}'.")
        print("Available scripts: train_vqvae, train_vqvae_cnn")
        sys.exit(1)

    script_path = scripts[script_name]

    # Run the selected script
    subprocess.run(["python", script_path])

if __name__ == "__main__":
    main()