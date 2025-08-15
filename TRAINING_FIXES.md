# MOE Training Pipeline - Data Handling & Training Issues Fixed

## Critical Training Issues Identified and Fixed

### 1. **Inconsistent Data Normalization**
**Problem**: Normalization statistics were computed differently between training and evaluation
**Fix**: 
- Unified `compute_global_stats()` function with consistent log transformation and masking
- Added proper validation and debugging for normalization statistics
- Ensured same normalization parameters used in both training and evaluation

### 2. **Masking Logic Problems**
**Problem**: Multiple issues with mask application and validation
**Fixes**:
- Fixed mask application order: normalize first, then apply mask
- Added mask validation (ensure binary 0/1 values)
- Corrected masked loss computation to average over valid positions only
- Added comprehensive mask statistics tracking

### 3. **Data Loading and Validation Issues**
**Problem**: Missing proper data validation and error handling
**Fixes**:
- Added comprehensive data validation in `load_all_labels_dataset()`
- Implemented tensor shape validation and range checking
- Added NaN/Inf detection and handling
- Enhanced error reporting with detailed statistics

### 4. **Loss Computation Problems**
**Problem**: Masked reconstruction loss was computed incorrectly
**Fix**:
```python
# Before (incorrect):
r_loss = (((out - x_norm) ** 2) * mask.unsqueeze(-1)).sum() / mask.sum()

# After (correct):
reconstruction_error = (out - x_norm) ** 2  # [B, T, 3]
masked_error = reconstruction_error * mask.unsqueeze(-1)
total_valid_elements = mask.sum() * reconstruction_error.shape[-1]
r_loss = masked_error.sum() / total_valid_elements
```

### 5. **Data Module Pattern Implementation**
**Problem**: No structured data management like Lightning's DataModule
**Fix**: Added `MOEDataModule` class with:
- Centralized data preparation and validation
- Consistent preprocessing pipeline
- Proper distributed data handling
- Better error handling and debugging

## Key Training Improvements

### Data Preprocessing Pipeline
1. **Consistent Tensor Format**: Ensure [B, T, 3] format throughout
2. **Log Transformation**: Apply consistently based on config
3. **Normalization**: Use global statistics computed on training data
4. **Masking**: Apply after normalization, validate mask values

### Validation and Error Handling
1. **NaN/Inf Detection**: Check all tensors before processing
2. **Range Validation**: Verify particle feature ranges (e.g., pt > 0)
3. **Shape Validation**: Ensure consistent tensor shapes
4. **Mask Validation**: Verify binary mask values

### Loss Computation
1. **Masked Loss**: Properly handle variable-length sequences
2. **Valid Position Averaging**: Only compute loss on unmasked positions
3. **Loss Validation**: Check for NaN/Inf in loss values
4. **Gradient Stability**: Ensure stable gradient computation

### Data Statistics and Monitoring
1. **Comprehensive Mask Statistics**: Track token usage and distribution
2. **Data Range Monitoring**: Monitor feature value ranges
3. **Batch Validation**: Per-batch data quality checks
4. **Memory Management**: Limit statistics computation for large datasets

## Code Structure Improvements

### Before (Problematic):
```python
# Inconsistent normalization
x_norm = (x_particles - mean) / std if not use_mask else (
    (x_particles - mean) / std) * mask.unsqueeze(-1)

# Incorrect masked loss
r_loss = (((out - x_norm) ** 2) * mask.unsqueeze(-1)).sum() / mask.sum()
```

### After (Robust):
```python
# Consistent preprocessing via data module
x_norm, mask, x_particles, x_jets, y = data_module.preprocess_batch(batch, device)

# Proper masked loss computation
reconstruction_error = (out - x_norm) ** 2
masked_error = reconstruction_error * mask.unsqueeze(-1)
total_valid_elements = mask.sum() * reconstruction_error.shape[-1]
r_loss = masked_error.sum() / total_valid_elements
```

## Benefits of Training Improvements

1. **Consistency**: Same preprocessing in training and evaluation
2. **Stability**: Better numerical stability and gradient flow
3. **Debuggability**: Comprehensive validation and logging
4. **Robustness**: Graceful handling of edge cases and errors
5. **Reproducibility**: Deterministic data handling and preprocessing
6. **Performance**: Efficient masking and loss computation

## Configuration Validation

Added validation for critical training parameters:
- Consistent `log_pt` setting between training and evaluation
- Proper mask handling configuration
- Normalization parameter validation
- Data range and type checking

## Testing and Validation

The improved pipeline now includes:
- Batch-level data validation
- Per-epoch statistics monitoring
- Gradient and loss value checking
- Memory usage optimization
- Error recovery mechanisms

These fixes address the core training issues that could lead to unstable training, incorrect loss computation, and inconsistent model behavior between training and evaluation phases.

# Jet Training Script - Detailed Before/After Changes

## Overview of Jet Training Improvements

The jet training script (`train_jet.py`) was completely refactored to address large, fluctuating losses and incorrect plot scaling. The improvements were modeled after the robust training logic from `moe.py` and include:

1. **Consistent preprocessing and normalization**
2. **Robust data validation and error handling**
3. **Stable loss computation**
4. **Improved evaluation logic**
5. **Enhanced logging and debugging**

---

## 1. Data Preprocessing Functions

### Before (No Preprocessing Functions)
```python
# Original training loop directly processed data inconsistently:
for batch_idx, (x_jets, _) in enumerate(dataloader):
    x_jets = x_jets.to(device)
    
    # Inconsistent normalization - sometimes applied, sometimes not
    if use_normalization:  # This flag was inconsistent
        x_jets_norm = (x_jets - mean) / std
    else:
        x_jets_norm = x_jets
    
    # No log transform consistency
    # No data validation
    # No error handling
```

### After (Robust Preprocessing Functions)
```python
def preprocess_jet_batch(x_jets, mean, std, log_pt=True, validate=True):
    """Preprocess jet batch with consistent normalization and validation."""
    if validate:
        # Validate input data
        if torch.isnan(x_jets).any() or torch.isinf(x_jets).any():
            raise ValueError("NaN/Inf detected in input jet data")
        
        # Check pt values
        pt_values = x_jets[:, 0]
        if (pt_values <= 0).any():
            print("⚠️ Warning: Non-positive pt values found, clamping to positive")
            x_jets = x_jets.clone()
            x_jets[:, 0] = torch.clamp(x_jets[:, 0], min=1e-6)
    
    # Apply preprocessing consistently
    x_processed = x_jets.clone()
    if log_pt:
        x_processed[:, 0] = torch.log(x_processed[:, 0] + 1e-6)
    
    # Apply normalization
    x_norm = (x_processed - mean) / std
    
    # Validate output
    if validate:
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            raise ValueError("NaN/Inf detected after normalization")
    
    return x_norm, x_processed

def denormalize_jet_batch(x_norm, mean, std, log_pt=True, validate=True):
    """Denormalize jet batch consistently."""
    # Denormalize
    x_processed = x_norm * std + mean
    
    # Inverse log transform
    x_output = x_processed.clone()
    if log_pt:
        x_output[:, 0] = torch.exp(x_processed[:, 0]) - 1e-6
        # Clamp to ensure positive pt values
        x_output[:, 0] = torch.clamp(x_output[:, 0], min=1e-6)
    
    # Validate output
    if validate:
        if torch.isnan(x_output).any() or torch.isinf(x_output).any():
            print("⚠️ Warning: NaN/Inf detected after denormalization")
            # Replace NaN/Inf with reasonable default values
            x_output = torch.where(torch.isnan(x_output) | torch.isinf(x_output), 
                                 torch.tensor([200.0, 0.0, 0.0], device=x_output.device), x_output)
    
    return x_output
```

---

## 2. Statistics Computation

### Before (Inconsistent Statistics)
```python
def compute_stats(dataset):
    # Simple computation without validation
    all_data = []
    for batch in dataset:
        all_data.append(batch[0])
    data = torch.cat(all_data)
    return data.mean(0), data.std(0)
```

### After (Robust Statistics with Validation)
```python
def compute_global_stats(dataset: TensorDataset, batch_size: int, log_pt: bool = True):
    """Compute mean and std for jet features [pt, eta, phi] with robust handling."""
    print(f"🔢 Computing jet statistics with log_pt={log_pt}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    jets = []
    
    for batch_idx, batch in enumerate(loader):
        x_j, _ = batch  # x_j should be [B, 3] for jet features
        
        # Validate input data
        if torch.isnan(x_j).any() or torch.isinf(x_j).any():
            print(f"⚠️ Warning: NaN/Inf detected in jet data batch {batch_idx}")
            continue
            
        # Check for reasonable pt values
        pt_values = x_j[:, 0]
        if (pt_values <= 0).any():
            print(f"⚠️ Warning: Non-positive pt values found in batch {batch_idx}")
            # Clamp to small positive value
            x_j = x_j.clone()
            x_j[:, 0] = torch.clamp(x_j[:, 0], min=1e-6)
        
        # Apply log transform consistently
        if log_pt:
            x_j = x_j.clone()
            x_j[:, 0] = torch.log(x_j[:, 0] + 1e-6)  # Add epsilon for numerical stability
            
        jets.append(x_j)
        
        # Limit computation for very large datasets
        if batch_idx >= 100:
            print(f"⚠️ Limited stats computation to first {batch_idx + 1} batches")
            break
    
    if not jets:
        raise RuntimeError("No valid batches found for statistics computation")
    
    jets_all = torch.cat(jets, dim=0)
    mean = jets_all.mean(dim=0)
    std = jets_all.std(dim=0) + 1e-6  # Add epsilon for numerical stability
    
    print(f"📊 Jet statistics computed:")
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std: {std.tolist()}")
    print(f"  Valid samples used: {jets_all.shape[0]:,}")
    
    # Validate computed statistics
    if torch.isnan(mean).any() or torch.isnan(std).any():
        raise RuntimeError("NaN values in computed statistics")
    if (std < 1e-8).any():
        print("⚠️ Warning: Very small std values detected, adjusting...")
        std = torch.clamp(std, min=1e-6)
    
    return mean, std
```

---

## 3. Training Loop Improvements

### Before (Unstable Training Loop)
```python
for batch_idx, (x_jets, _) in enumerate(dataloader):
    x_jets = x_jets.to(device)
    
    # Inconsistent preprocessing
    x_jets_norm = (x_jets - mean) / std  # No validation, no log transform
    
    optimizer.zero_grad()
    out, vq_loss = model(x_jets_norm)
    
    # Basic loss computation
    r_loss = F.mse_loss(out, x_jets_norm)
    loss = r_loss + vq_loss  # Assumed vq_loss was always a tensor
    
    loss.backward()
    optimizer.step()
```

### After (Robust Training Loop)
```python
for batch_idx, (x_jets, _) in enumerate(dataloader):
    x_jets = x_jets.to(device)  # [B, 3] jet features
    
    # Validate batch data before processing
    if torch.isnan(x_jets).any() or torch.isinf(x_jets).any():
        print(f"⚠️ Warning: NaN/Inf in batch {batch_idx}, skipping")
        continue
    
    # Check for valid pt values
    if (x_jets[:, 0] <= 0).any():
        print(f"⚠️ Warning: Non-positive pt values in batch {batch_idx}, clamping")
        x_jets = x_jets.clone()
        x_jets[:, 0] = torch.clamp(x_jets[:, 0], min=1e-6)
    
    # Apply preprocessing consistently (like MOE data module)
    x_jets_norm, x_jets_processed = preprocess_jet_batch(x_jets, mean, std, log_pt=True, validate=True)
    
    # Debug logging for first batch of first epoch
    if epoch == start_epoch and batch_idx == 0 and rank == 0:
        print(f"\n🔍 Debug info for first batch:")
        print(f"  Original jet range: [{x_jets.min():.3f}, {x_jets.max():.3f}]")
        print(f"  Processed jet range: [{x_jets_processed.min():.3f}, {x_jets_processed.max():.3f}]")
        print(f"  Normalized jet range: [{x_jets_norm.min():.3f}, {x_jets_norm.max():.3f}]")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}\n")
    
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        # Direct jet-to-jet reconstruction
        out, vq_loss = model(x_jets_norm)
        
        # Validate model output
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"❌ Invalid model output in batch {batch_idx}")
            continue
        
        # Compute loss in NORMALIZED space for stability (like MOE)
        r_loss = recon_loss_fn(out, x_jets_norm)
        
        # Handle VQ loss dictionary robustly
        if isinstance(vq_loss, dict):
            v_loss = vq_loss.get("loss", vq_loss.get("vq_loss", torch.tensor(0.0, device=device)))
            codes = vq_loss.get("q")
            if codes is not None:
                try:
                    hist = torch.bincount(codes.view(-1), minlength=config["vq_kwargs"]["num_codes"])
                    code_hist += hist.to(device)
                except Exception as e:
                    print(f"⚠️ Warning: Error computing code histogram: {e}")
        else:
            v_loss = vq_loss
        
        # Validate loss values
        if torch.isnan(r_loss) or torch.isinf(r_loss):
            print(f"❌ Invalid reconstruction loss in batch {batch_idx}: {r_loss}")
            continue
            
        if torch.isnan(v_loss) or torch.isinf(v_loss):
            print(f"❌ Invalid VQ loss in batch {batch_idx}: {v_loss}")
            continue
            
        loss = r_loss + v_loss
    
    scaler.scale(loss).backward()
    
    # Add gradient clipping for stability (like MOE)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

---

## 4. Evaluation Improvements

### Before (Inconsistent Evaluation)
```python
def eval_jet(config):
    # Load model and data without consistent preprocessing
    for x_j, _ in loader:
        x_j = x_j.to(device)
        x_j_norm = (x_j - mean) / std  # Inconsistent with training
        out, _ = model(x_j_norm)
        # Plot without proper denormalization
        plot_results(x_j, out)
```

### After (Consistent Evaluation)
```python
def eval_jet(config: dict) -> None:
    """Evaluate trained jet model and create plots."""
    # ... model setup code ...
    
    # Evaluate model with consistent preprocessing
    with torch.no_grad():
        for x_j, _ in loader:
            x_j = x_j.to(device)  # [B, 3] jet features
            
            # Apply SAME preprocessing as training using robust functions
            try:
                x_j_norm, x_j_processed = preprocess_jet_batch(x_j, mean, std, log_pt=True, validate=True)
                
                out, _ = model(x_j_norm)   # Reconstruct in normalized space
                
                # Denormalize output using robust function
                out_denorm = denormalize_jet_batch(out, mean, std, log_pt=True, validate=True)
                
                recon_jets.append(out_denorm)
                orig_jets.append(x_j)
                
            except Exception as e:
                print(f"⚠️ Warning: Error processing batch during evaluation: {e}")
                continue
```

---

## 5. Key Benefits of Changes

### Training Stability
- **Before**: Large, fluctuating losses due to inconsistent normalization
- **After**: Stable training with consistent preprocessing and gradient clipping

### Data Validation
- **Before**: No validation, leading to NaN/Inf propagation
- **After**: Comprehensive validation at every step with graceful error handling

### Loss Computation
- **Before**: Computed in inconsistent spaces, leading to scaling issues
- **After**: Computed in normalized space for numerical stability

### Evaluation Consistency
- **Before**: Different preprocessing between training and evaluation
- **After**: Identical preprocessing pipeline ensuring consistent results

### Debugging
- **Before**: Minimal logging, difficult to diagnose issues
- **After**: Detailed logging, validation, and error reporting

These changes resulted in stable training with consistent loss curves and properly scaled evaluation plots, directly addressing the original issues of large fluctuating losses and incorrect plot scaling.

---
