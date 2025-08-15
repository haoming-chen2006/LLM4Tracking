# Dataloader Documentation

This document describes the data loading infrastructure for the HEP models project. The dataloader components handle reading, preprocessing, and batching of the JetClass dataset for training and evaluation.

## Overview

The dataloader system is designed around the **JetClass dataset**, which contains high-energy physics collision events with labeled jet types. The system provides flexible data loading with support for:

- Multiple jet class labels
- Variable-length particle sequences
- Masking for padded sequences
- Distributed training
- Efficient batching and preprocessing

## Dataset Structure

### JetClass Dataset
- **Format**: ROOT files with particle and jet features
- **Particle Features**: `[pt, eta, phi]` for each constituent
- **Jet Features**: `[pt, eta, phi, mass]` for the whole jet
- **Labels**: 10 different jet types (HToXX, ZToXX, etc.)
- **Size**: ~100M jets across different physics processes

### Data Organization
```
data/
├── jetclass/
│   ├── HToBB/           # Higgs to bottom-antibottom
│   ├── HToCC/           # Higgs to charm-anticharm  
│   ├── HToGG/           # Higgs to gluon-gluon
│   ├── HToWW4Q/         # Higgs to W+W- (4 quarks)
│   ├── HToWW2Q1L/       # Higgs to W+W- (2 quarks, 1 lepton)
│   ├── ZToQQ/           # Z boson to quark-antiquark
│   ├── WToQQ/           # W boson to quark-antiquark
│   ├── TTBar/           # Top-antitop pairs
│   ├── TTBarLep/        # Top-antitop with leptons
│   └── ZJetsToNuNu/     # Z + jets to neutrinos
```

## Core Components

### 1. Base Dataloader - `dataloader/dataloader.py`

**Purpose**: Standard dataloader without masking support

**Key Functions:**
- `read_file(file_path, start, end)`: Read ROOT files and extract features
- `load_jetclass_label_as_tensor()`: Load specific jet class as DataLoader
- `load_jetclass_label_as_dataset()`: Load specific jet class as TensorDataset

**Output Format:**
```python
# Returns tuple of tensors
x_particles: torch.Tensor  # [batch_size, 3, max_particles] - particle features
x_jets: torch.Tensor       # [batch_size, 4] - jet features  
y: torch.Tensor           # [batch_size] - class labels
```

**Usage Example:**
```python
from dataloader.dataloader import load_jetclass_label_as_tensor

# Load HToBB jets from files 0-5
dataloader = load_jetclass_label_as_tensor(
    label="HToBB",
    start=0,
    end=5,
    batch_size=512
)

for x_particles, x_jets, y in dataloader:
    # x_particles: [512, 3, 128] - up to 128 particles per jet
    # x_jets: [512, 4] - jet-level features
    # y: [512] - all HToBB labels
    process_batch(x_particles, x_jets, y)
```

**Features:**
- Fixed-size particle sequences (padded with zeros)
- Fast loading for models that don't need masking
- Memory efficient for dense sequences
- Compatible with all PyTorch utilities

### 2. Masked Dataloader - `dataloader/masked_dataloader.py`

**Purpose**: Advanced dataloader with masking for variable-length sequences

**Key Functions:**
- `read_file_with_mask()`: Read files and generate validity masks
- `load_jetclass_label_as_tensor()`: Load with mask support
- `load_jetclass_label_as_dataset()`: Dataset version with masks

**Output Format:**
```python
# Returns tuple with additional mask
x_particles: torch.Tensor  # [batch_size, 3, max_particles] - particle features
x_jets: torch.Tensor       # [batch_size, 4] - jet features
y: torch.Tensor           # [batch_size] - class labels
mask: torch.Tensor        # [batch_size, max_particles] - validity mask (1=valid, 0=padding)
```

**Usage Example:**
```python
from dataloader.masked_dataloader import load_jetclass_label_as_tensor

# Load with masking support
dataloader = load_jetclass_label_as_tensor(
    label="ZToQQ", 
    start=10, 
    end=15,
    batch_size=256
)

for x_particles, x_jets, y, mask in dataloader:
    # mask: [256, 128] - indicates which particles are real vs padding
    valid_particles = x_particles * mask.unsqueeze(1)  # Apply mask
    process_variable_length_batch(valid_particles, mask)
```

**Features:**
- Variable-length sequence support
- Memory efficient for sparse sequences
- Essential for Flash and MOE models
- Proper gradient computation only on valid tokens

### 3. Configuration Loader - `dataloader/load.py`

**Purpose**: High-level interface using configuration files

**Configuration Format** (`dataloader/config.yaml`):
```yaml
data_path: "/path/to/jetclass"
labels:
  - HToBB
  - HToCC
  - HToGG
  # ... more labels
train_files: [0, 5]    # File range for training
val_files: [5, 8]      # File range for validation
test_files: [8, 10]    # File range for testing
```

**Usage:**
```python
from dataloader.load import load_config, create_dataloaders

config = load_config("dataloader/config.yaml")
train_loader, val_loader, test_loader = create_dataloaders(config)
```

### 4. Statistics Computer - `dataloader/mean_std.py`

**Purpose**: Compute normalization statistics for training

**Key Functions:**
- `compute_mean_std()`: Calculate dataset statistics
- `normalize_particles()`: Apply normalization to particle features
- `denormalize_particles()`: Reverse normalization for reconstruction

**Usage:**
```python
from dataloader.mean_std import compute_mean_std

# Compute statistics on training data
mean, std = compute_mean_std(train_dataset, use_log_pt=False)

# Apply during training
x_norm = (x_particles - mean) / std
```

## Data Preprocessing Pipeline

### 1. File Reading
```python
def read_file(file_path, start, end):
    """
    Read ROOT file and extract features
    
    Returns:
        particles: numpy.ndarray [N_jets, N_particles, 3]
        jets: numpy.ndarray [N_jets, 4] 
        labels: numpy.ndarray [N_jets]
    """
```

### 2. Padding and Batching
- Particles padded to maximum sequence length (default: 128)
- Zero-padding for shorter sequences
- Masking indicates valid vs padded particles

### 3. Tensor Conversion
- Convert NumPy arrays to PyTorch tensors
- Proper device placement (CPU/GPU)
- Data type consistency (float32)

### 4. Normalization (Optional)
```python
# Log-transform pt values (for some models)
if use_log_pt:
    x_particles[:, 0] = torch.log(x_particles[:, 0] + 1e-6)

# Z-score normalization
x_normalized = (x_particles - mean) / std
```

## Advanced Features

### 1. Multi-Label Loading
```python
from dataloader.dataloader import load_all_labels_dataset

# Load multiple jet types together
dataset = load_all_labels_dataset(
    labels=["HToBB", "HToCC", "ZToQQ"],
    start=0,
    end=10,
    use_mask=True
)
```

### 2. Distributed Loading
```python
from torch.utils.data import DistributedSampler

# For multi-GPU training
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```

### 3. Custom Collate Functions
```python
def custom_collate_fn(batch):
    """Custom batching for complex data structures"""
    particles, jets, labels, masks = zip(*batch)
    
    # Custom padding/masking logic
    particles = pad_sequence(particles, batch_first=True)
    masks = create_attention_masks(particles)
    
    return particles, jets, labels, masks
```

### 4. Memory-Efficient Loading
```python
# Use pin_memory for faster GPU transfer
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=2
)
```

## Performance Optimization

### 1. Caching
- Pre-computed statistics cached to disk
- File metadata cached for faster startup
- Processed tensors can be cached for repeated experiments

### 2. Parallel Loading
- Multi-worker data loading (num_workers=4-8)
- Prefetching to overlap data loading with training
- Pin memory for faster GPU transfers

### 3. Memory Management
```python
# Memory-efficient loading for large datasets
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,      # Avoid small final batches
    prefetch_factor=2    # Reduce memory usage
)
```

## Common Use Cases

### 1. Single Label Training
```python
# Train on specific jet type
train_loader = load_jetclass_label_as_tensor("HToBB", 0, 10, batch_size=512)
val_loader = load_jetclass_label_as_tensor("HToBB", 10, 12, batch_size=512)
```

### 2. Multi-Label Classification
```python
# Train classifier on all jet types  
all_labels = ["HToBB", "HToCC", "HToGG", "ZToQQ", "WToQQ"]
dataset = load_multiple_labels(all_labels, 0, 50)
```

### 3. Reconstruction Training
```python
# VQ-VAE training (no labels needed)
for x_particles, x_jets, _ in dataloader:
    reconstruction = model(x_particles)
    loss = mse_loss(reconstruction, x_particles)
```

### 4. Few-Shot Learning
```python
# Small dataset for quick experiments
quick_loader = load_jetclass_label_as_tensor("HToBB", 0, 1, batch_size=64)
```

## Configuration Examples

### Basic Configuration
```python
BATCH_SIZE = 512
MAX_PARTICLES = 128
USE_MASKING = True
NORMALIZE = True
LOG_PT = False  # Usually False for MOE models
```

### Memory-Limited Setup
```python
BATCH_SIZE = 256        # Reduce for limited memory
MAX_PARTICLES = 64      # Shorter sequences
NUM_WORKERS = 2         # Fewer worker processes
PIN_MEMORY = False      # Reduce memory usage
```

### High-Performance Setup
```python
BATCH_SIZE = 1024       # Large batches for efficiency
NUM_WORKERS = 8         # More parallel loading
PIN_MEMORY = True       # Faster GPU transfer
PREFETCH_FACTOR = 4     # More prefetching
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch_size or max_particles
   - Decrease num_workers
   - Disable pin_memory

2. **Slow Loading**
   - Increase num_workers (up to CPU count)
   - Enable pin_memory
   - Use SSD storage for data files

3. **Inconsistent Batch Sizes**
   - Set drop_last=True
   - Check file boundaries align with batch_size

4. **Masking Issues**
   - Ensure mask dimensions match particle dimensions
   - Verify mask values are 0/1 only
   - Check that masked positions have zero gradients

### Debug Utilities
```python
# Check data statistics
def check_dataloader(dataloader):
    for batch in dataloader:
        if len(batch) == 4:  # Masked loader
            x_particles, x_jets, y, mask = batch
            print(f"Particles: {x_particles.shape}, Mask: {mask.shape}")
            print(f"Mask ratio: {mask.float().mean():.3f}")
        else:  # Standard loader  
            x_particles, x_jets, y = batch
            print(f"Particles: {x_particles.shape}")
        
        print(f"Particle ranges: pt [{x_particles[:,:,0].min():.2f}, {x_particles[:,:,0].max():.2f}]")
        break
```

This dataloader system provides a robust foundation for training various types of models on the JetClass dataset, with flexibility for different experimental needs and computational constraints.
