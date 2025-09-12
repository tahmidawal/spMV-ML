#!/usr/bin/env python3
"""
Verify if the Two-Matrix model weights are actually memorizing or just random noise
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from train_dense_spmv import DenseSpMV_TwoMatrix
import os

def verify_memorization():
    print("=" * 80)
    print("MEMORIZATION VERIFICATION TEST")
    print("=" * 80)
    
    # Load the trained model
    checkpoint = torch.load('dense_spmv_results_20250911_144313/two_matrix_2x324_best.pth', map_location='cpu')
    model = DenseSpMV_TwoMatrix(vector_dim=648, matrix_shape=(2, 324))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load training and validation data
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    X_train = torch.FloatTensor(data['X_train'])
    Y_train = torch.FloatTensor(data['Y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    Y_val = torch.FloatTensor(data['Y_val'])
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Extract learned weights
    W1 = model.W1.detach().numpy()
    W2 = model.W2.detach().numpy()
    bias = model.bias.detach().numpy()
    
    print(f"\nModel weights:")
    print(f"W1 shape: {W1.shape}, range: [{W1.min():.6f}, {W1.max():.6f}]")
    print(f"W2 shape: {W2.shape}, range: [{W2.min():.6f}, {W2.max():.6f}]")
    print(f"Bias range: [{bias.min():.6f}, {bias.max():.6f}]")
    
    # Test 1: Performance on training vs validation
    print("\n" + "="*50)
    print("TEST 1: TRAINING vs VALIDATION PERFORMANCE")
    print("="*50)
    
    with torch.no_grad():
        # Training performance
        train_mse = []
        for i in range(X_train.shape[0]):
            pred = model(X_train[i:i+1]).squeeze()
            mse = ((pred - Y_train[i]) ** 2).mean().item()
            train_mse.append(mse)
        
        # Validation performance
        val_mse = []
        for i in range(X_val.shape[0]):
            pred = model(X_val[i:i+1]).squeeze()
            mse = ((pred - Y_val[i]) ** 2).mean().item()
            val_mse.append(mse)
    
    print(f"Training MSE: mean={np.mean(train_mse):.2e}, std={np.std(train_mse):.2e}")
    print(f"Validation MSE: mean={np.mean(val_mse):.2e}, std={np.std(val_mse):.2e}")
    print(f"Ratio (val/train): {np.mean(val_mse)/np.mean(train_mse):.2f}")
    
    # Test 2: Sensitivity to weight perturbation
    print("\n" + "="*50)
    print("TEST 2: SENSITIVITY TO WEIGHT PERTURBATION")
    print("="*50)
    
    # Test small perturbations
    perturbations = [0.001, 0.01, 0.1]
    
    for pert in perturbations:
        # Perturb W2 (the big matrix)
        W2_perturbed = model.W2.data + torch.randn_like(model.W2) * pert
        original_W2 = model.W2.data.clone()
        
        model.W2.data = W2_perturbed
        
        with torch.no_grad():
            perturbed_mse = []
            for i in range(min(10, X_val.shape[0])):  # Test on first 10 samples
                pred = model(X_val[i:i+1]).squeeze()
                mse = ((pred - Y_val[i]) ** 2).mean().item()
                perturbed_mse.append(mse)
        
        # Restore original weights
        model.W2.data = original_W2
        
        print(f"Perturbation {pert}: MSE = {np.mean(perturbed_mse):.2e}")
    
    # Test 3: Random input test
    print("\n" + "="*50)
    print("TEST 3: RANDOM INPUT RESPONSE")
    print("="*50)
    
    # Generate random inputs
    random_inputs = torch.randn(5, 648)
    
    with torch.no_grad():
        for i, rand_input in enumerate(random_inputs):
            output = model(rand_input.unsqueeze(0)).squeeze()
            print(f"Random input {i}: range=[{rand_input.min():.3f}, {rand_input.max():.3f}] "
                  f"-> output=[{output.min():.3f}, {output.max():.3f}]")
    
    # Test 4: Weight matrix structure analysis
    print("\n" + "="*50)
    print("TEST 4: WEIGHT MATRIX STRUCTURE")
    print("="*50)
    
    # Analyze W2 structure
    print("W2 Matrix Analysis:")
    print(f"  Rank: {np.linalg.matrix_rank(W2)}/324")
    print(f"  Condition number: {np.linalg.cond(W2):.2e}")
    
    # Check if W2 has block structure
    U, s, Vt = np.linalg.svd(W2)
    effective_rank = np.sum(s > 0.01 * s[0])
    print(f"  Effective rank (1% threshold): {effective_rank}")
    print(f"  Top 10 singular values: {s[:10]}")
    
    # Test 5: Interpolation test
    print("\n" + "="*50)
    print("TEST 5: INTERPOLATION BETWEEN TRAINING SAMPLES")
    print("="*50)
    
    # Take two training samples and interpolate
    idx1, idx2 = 0, 1
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    with torch.no_grad():
        for alpha in alphas:
            # Interpolate inputs
            x_interp = (1 - alpha) * X_train[idx1] + alpha * X_train[idx2]
            y_true_interp = (1 - alpha) * Y_train[idx1] + alpha * Y_train[idx2]  # Linear interpolation
            
            # Model prediction
            y_pred = model(x_interp.unsqueeze(0)).squeeze()
            
            mse = ((y_pred - y_true_interp) ** 2).mean().item()
            print(f"Alpha={alpha:.2f}: MSE={mse:.2e}")
    
    return model, W1, W2, bias

if __name__ == "__main__":
    model, W1, W2, bias = verify_memorization()