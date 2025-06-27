import torch
import psutil
import os

def print_memory_stats():
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB")
    print(f"Available: {memory.available / 1024**3:.2f} GB")

if __name__ == "__main__":
    print_memory_stats()
