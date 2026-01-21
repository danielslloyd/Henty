#!/usr/bin/env python3
"""
Quick GPU/CUDA diagnostic script for Henty
Run this to check if PyTorch can see your GPU
"""

import torch
import sys

print("="*80)
print("GPU/CUDA DIAGNOSTIC")
print("="*80)

# PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"PyTorch CUDA compiled: {torch.version.cuda}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    print("\n✓ GPU IS AVAILABLE!")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  Current GPU: {torch.cuda.current_device()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test tensor creation
    print("\nTesting GPU tensor creation...")
    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"✓ Successfully created tensor on GPU: {test_tensor.device}")
    except Exception as e:
        print(f"✗ Failed to create tensor on GPU: {e}")
else:
    print("\n✗ GPU IS NOT AVAILABLE")
    print("\nPossible issues:")
    print("  1. PyTorch installed without CUDA support")
    print("  2. NVIDIA GPU drivers not installed")
    print("  3. CUDA toolkit not installed or incompatible version")

    print("\nDiagnostic steps:")
    print("  1. Check GPU is detected:")
    print("     $ nvidia-smi")
    print("     (Should show your GPU and driver version)")

    print("\n  2. Check PyTorch installation:")
    print("     $ python -c 'import torch; print(torch.__version__)'")
    print("     (Should show version with +cu118 or similar for CUDA)")

    print("\n  3. Reinstall PyTorch with CUDA support:")
    print("     Visit: https://pytorch.org/get-started/locally/")
    print("     Select your system, CUDA version, and copy the install command")
    print("     Example for CUDA 11.8:")
    print("     $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*80)

sys.exit(0 if cuda_available else 1)
