import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """Format numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def main():
    print("🔍 Model Parameter Analysis")
    print("=" * 80)
    
    # Model configurations from the scripts
    models_config = {
        "NormFormer (Original)": {
            "module": "models.NormFormer",
            "params": {
                "input_dim": 3,
                "latent_dim": 128,
                "hidden_dim": 256,
                "num_heads": 8,
                "num_blocks": 3,
                "vq_kwargs": {"num_codes": 2048, "beta": 0.25}
            }
        },
        "NormFormer Flash": {
            "module": "models.NormFormer_Flash",
            "params": {
                "input_dim": 3,
                "latent_dim": 16,
                "hidden_dim": 128,
                "num_heads": 8,
                "num_blocks": 3,
                "vq_kwargs": {"num_codes": 2048, "beta": 0.25}
            }
        },
        "MOE Medium": {
            "module": "models.MOE",
            "params": {
                "input_dim": 3,
                "latent_dim": 16,
                "hidden_dim": 128,
                "num_heads": 8,
                "num_blocks": 3,
                "vq_kwargs": {"num_codes": 4096, "beta": 0.8}
            }
        },
        "MOE Large": {
            "module": "models.MOE",
            "params": {
                "input_dim": 3,
                "latent_dim": 16,
                "hidden_dim": 128,
                "num_heads": 8,
                "num_blocks": 3,
                "vq_kwargs": {"num_codes": 8192, "beta": 0.9}
            }
        }
    }
    
    results = []
    
    for model_name, config in models_config.items():
        try:
            print(f"\n📊 Loading {model_name}...")
            
            # Import the model module
            model_module = __import__(config["module"], fromlist=["VQVAENormFormer"])
            
            # Create the model
            model = model_module.VQVAENormFormer(**config["params"])
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            
            # Get VQ layer info
            vq_codes = config["params"]["vq_kwargs"]["num_codes"]
            latent_dim = config["params"]["latent_dim"]
            codebook_params = vq_codes * latent_dim
            
            results.append({
                "name": model_name,
                "total": total_params,
                "trainable": trainable_params,
                "codebook": codebook_params,
                "non_codebook": total_params - codebook_params
            })
            
            print(f"✅ {model_name}")
            print(f"   Total parameters: {format_number(total_params)} ({total_params:,})")
            print(f"   Trainable: {format_number(trainable_params)} ({trainable_params:,})")
            print(f"   Codebook: {format_number(codebook_params)} ({codebook_params:,})")
            print(f"   Non-codebook: {format_number(total_params - codebook_params)} ({total_params - codebook_params:,})")
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    # Summary table
    print("\n" + "=" * 80)
    print("📋 PARAMETER SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Total':<12} {'Trainable':<12} {'Codebook':<12} {'Non-Codebook':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<25} "
              f"{format_number(result['total']):<12} "
              f"{format_number(result['trainable']):<12} "
              f"{format_number(result['codebook']):<12} "
              f"{format_number(result['non_codebook']):<15}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("📈 ANALYSIS")
    print("=" * 80)
    
    if len(results) >= 2:
        # Compare models
        smallest = min(results, key=lambda x: x['total'])
        largest = max(results, key=lambda x: x['total'])
        
        print(f"💡 Smallest model: {smallest['name']} with {format_number(smallest['total'])} parameters")
        print(f"💡 Largest model: {largest['name']} with {format_number(largest['total'])} parameters")
        
        if largest['total'] > smallest['total']:
            ratio = largest['total'] / smallest['total']
            print(f"💡 Size ratio (largest/smallest): {ratio:.2f}x")
        
        # Codebook analysis
        codebook_sizes = [(r['name'], r['codebook']) for r in results]
        codebook_sizes.sort(key=lambda x: x[1])
        
        print(f"\n🔖 Codebook sizes:")
        for name, size in codebook_sizes:
            print(f"   {name}: {format_number(size)} parameters")
        
        # Model efficiency (non-codebook params)
        print(f"\n🏗️  Model architectures (excluding codebook):")
        for result in results:
            non_cb = result['non_codebook']
            total = result['total']
            cb_ratio = (result['codebook'] / total) * 100
            print(f"   {result['name']}: {format_number(non_cb)} ({cb_ratio:.1f}% is codebook)")

if __name__ == "__main__":
    main()
