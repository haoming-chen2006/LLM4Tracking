# Model Documentation

This document provides detailed information about the models implemented in the HEP models project. The models are of two major classes -- VQVAE encoders that learns the jet/particle distribution, and bakcbones that integrate the vqvae with a pretrained language model, usually NanoGPT or TinyLLama.


### 1. VQ-VAE MLP Models
All includes an encoder, a decoder, and a vectorquant based quantizer in the middle. With extra normalization layer (norm_former) and mixture of expert (MOE) applied to specific variations.


### 2. NormFormer Models

#### VQ-VAE NormFormer - `NormFormer.py`
- **Purpose**: Sequence modeling of jet constituents using attention
- **Architecture**: Stack of NormFormer blocks (LayerNorm + Self-Attention + MLP)
- **Input**: Sequences of particles `[B, T, 3]` where T is max particles
- **Output**: Reconstructed particle sequences
- **Use Case**: Medium-scale experiments with attention mechanisms

**Key Features:**
- Self-attention for particle interactions
- Positional encoding support
- Residual connections
- Configurable depth and width

**Architecture Details:**
```
Input [B, T, 3] → Linear Projection → [B, T, hidden_dim]
                      ↓
NormFormer Block 1: LayerNorm → MultiHeadAttention → LayerNorm → MLP
                      ↓
NormFormer Block 2: LayerNorm → MultiHeadAttention → LayerNorm → MLP
                      ↓
                    ... (repeat)
                      ↓
Vector Quantization → Decoder → Output [B, T, 3]
```

**Configuration:**
```python
model = VQVAENormFormer(
    input_dim=3,
    latent_dim=128,
    hidden_dim=256,
    num_heads=8,
    num_blocks=3,
    vq_kwargs={
        "num_codes": 2048,
        "beta": 0.25,
        "affine_lr": 0.0,
        "sync_nu": 2,
        "replace_freq": 20,
        "dim": -1
    }
)
```

#### VQ-VAE Flash - `NormFormer_Flash.py`
- **Purpose**: Scalable attention-based model with masking support
- **Architecture**: Enhanced NormFormer with FlashAttention optimization
- **Input**: Variable-length particle sequences with masks
- **Output**: Reconstructed particles matching input lengths
- **Use Case**: Large-scale training with variable-length jets

**Key Features:**
- FlashAttention for memory efficiency
- Native masking support for variable-length sequences
- Deeper architectures (up to 6+ blocks)
- Optimized for GPU training

**Masking Support:**
- Input masks `[B, T]` indicate valid particles (1) vs padding (0)
- Attention masks prevent attending to padding tokens
- Loss computation only on valid particles
- Memory efficient for sparse sequences

#### VQ-VAE MOE - `MOE.py`
- **Purpose**: Mixture of Experts for scalable and specialized modeling
- **Architecture**: NormFormer backbone with MoE layers
- **Input**: Particle sequences (with optional masking)
- **Output**: Reconstructed particles with expert routing information
- **Use Case**: Large-scale experiments requiring model capacity scaling

**Key Features:**
- Multiple expert networks (typically 4-8 experts)
- Top-k routing (usually k=2)
- Load balancing across experts
- Scalable to very large codebooks (8K+ codes)
- Expert specialization for different jet types

**MoE Architecture:**
```
Input → NormFormer Blocks → MoE Layer → More NormFormer → VQ → Decoder
                              ↓
                    Router → Expert 1, Expert 2, ..., Expert N
                              ↓
                    Weighted combination → Output
```

**Expert Routing:**
- Gating network routes tokens to top-k experts
- Load balancing ensures even expert utilization
- Auxiliary losses encourage expert diversity

**Configuration:**
```python
model = VQVAENormFormer(
    input_dim=3,
    latent_dim=16,        # Smaller for MoE
    hidden_dim=128,       # Efficient per-expert computation
    num_heads=8,
    num_blocks=3,
    vq_kwargs={
        "num_codes": 8192,     # Large codebook
        "beta": 0.9,           # Higher VQ weight
        "affine_lr": 1.0,      # Learnable affine transforms
        "sync_nu": 5,          # More frequent synchronization
        "replace_freq": 2,     # Aggressive code replacement
        "dim": -1
    }
)
```

### 3. Specialized Components

#### Vector Quantization - `vectorquant.py`
- **Purpose**: Discrete latent space learning
- **Features**: 
  - Exponential moving averages for codebook updates
  - Code replacement for unused vectors
  - Affine transformations for improved expressivity
  - Distributed training support

#### Backbones - `backbones.py`
- **Purpose**: Shared architectural components
- **Components**:
  - Multi-head attention layers
  - MLP blocks with various activations
  - Normalization layers (LayerNorm, RMSNorm)
  - Positional encoding schemes

## Training Considerations

### Memory Usage
- **MLP Models**: Very low memory (~1GB for training)
- **NormFormer**: Moderate memory (~4-8GB depending on sequence length)
- **Flash**: Optimized memory usage with FlashAttention
- **MOE**: High memory due to multiple experts (~16-32GB)

### Convergence Patterns
- **MLP**: Fast convergence (10-50 epochs)
- **NormFormer**: Medium convergence (50-200 epochs)
- **Flash**: Similar to NormFormer but more stable
- **MOE**: Slower initial convergence but better final performance (100-500 epochs)

### Hyperparameter Sensitivity
- **VQ Beta**: Critical for reconstruction quality (0.25-1.0)
- **Learning Rate**: Models prefer lower LR (1e-4 to 1e-5)
- **Codebook Size**: Larger codebooks improve reconstruction but hurt compression
- **Expert Count (MOE)**: More experts improve capacity but increase complexity

## Model Selection Guide

### For Quick Experiments
- **VQ-VAE MLP (Particles)**: Fast prototyping, algorithm validation
- **VQ-VAE MLP (Jets)**: Global jet property studies

### For Research
- **VQ-VAE NormFormer**: Balanced performance and interpretability
- **VQ-VAE Flash**: Variable-length sequences, moderate scale

### For Production
- **VQ-VAE MOE**: Best performance, large-scale deployment
- **VQ-VAE Flash**: Good balance of performance and efficiency

## Common Issues and Solutions

### Training Instability
- **Symptom**: Loss spikes, codebook collapse
- **Solution**: Lower learning rate, increase VQ beta, use gradient clipping

### Poor Reconstruction
- **Symptom**: High MSE loss, blurry outputs
- **Solution**: Increase model capacity, larger codebook, longer training

### Codebook Underutilization
- **Symptom**: Many unused codes, low perplexity
- **Solution**: Increase code replacement frequency, lower beta, add code diversity losses

### Memory Issues
- **Symptom**: OOM errors during training
- **Solution**: Use Flash models, reduce batch size, enable gradient checkpointing

## Future Developments

### Planned Improvements
- Hierarchical vector quantization for better compression
- Cross-attention between different jet types
- Integration with language models for text-to-jet generation
- Causal modeling for event-level generation

### Research Directions
- Multi-scale tokenization (particle → jet → event)
- Physics-informed loss functions
- Uncertainty quantification in reconstructions
- Domain adaptation across different detectors
