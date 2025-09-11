#!/usr/bin/env python3
"""
Simple debug script to identify GNN training issues
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

# Load data
print("Loading data...")
data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')

X_train = data['X_train'][:10]  # Just 10 samples for debugging
Y_train = data['Y_train'][:10]

print(f"Data loaded: {X_train.shape}, {Y_train.shape}")

# Convert to graph format
edge_index, edge_weight = from_scipy_sparse_matrix(K)
edge_weight = edge_weight.float()

print(f"Original edge weight range: [{edge_weight.min():.3f}, {edge_weight.max():.3f}]")

# Try different normalizations
print("\n=== TESTING DIFFERENT NORMALIZATIONS ===")

# Test 1: No edge weight normalization
print("Test 1: Using raw edge weights")
x_test = torch.FloatTensor(X_train[0:1]).unsqueeze(-1)  # [1, 648, 1]
y_test = torch.FloatTensor(Y_train[0:1]).unsqueeze(-1)  # [1, 648, 1]

print(f"Input range: [{x_test.min():.6f}, {x_test.max():.6f}]")
print(f"Target range: [{y_test.min():.6f}, {y_test.max():.6f}]")

# Simple 1-layer GCN
class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(1, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)

model = SimpleGCN()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Test forward pass
print("\nTesting forward pass...")
try:
    with torch.no_grad():
        pred = model(x_test.squeeze(0), edge_index, edge_weight)
        print(f"Prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
        print(f"Prediction has NaN: {torch.isnan(pred).any()}")
        print("✅ Forward pass successful")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

# Test with normalized edge weights
print("\nTest 2: Normalized edge weights")
edge_weight_norm = edge_weight / edge_weight.std()
y_test_norm = y_test / edge_weight.std().numpy()

print(f"Normalized edge weight range: [{edge_weight_norm.min():.3f}, {edge_weight_norm.max():.3f}]")
print(f"Normalized target range: [{y_test_norm.min():.6f}, {y_test_norm.max():.6f}]")

try:
    with torch.no_grad():
        pred_norm = model(x_test.squeeze(0), edge_index, edge_weight_norm)
        print(f"Normalized prediction range: [{pred_norm.min():.6f}, {pred_norm.max():.6f}]")
        print(f"Normalized prediction has NaN: {torch.isnan(pred_norm).any()}")
        print("✅ Normalized forward pass successful")
except Exception as e:
    print(f"❌ Normalized forward pass failed: {e}")

# Test gradient computation
print("\nTest 3: Gradient computation")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

try:
    pred = model(x_test.squeeze(0), edge_index, edge_weight_norm)
    loss = criterion(pred.unsqueeze(0), y_test_norm)
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss has NaN: {torch.isnan(loss)}")
    
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any()
            print(f"  {name}: grad_norm = {grad_norm:.6f}, has_NaN = {has_nan}")
        else:
            print(f"  {name}: no gradient")
    
    print("✅ Gradient computation successful")
    
except Exception as e:
    print(f"❌ Gradient computation failed: {e}")

print("\nDEBUG COMPLETE")
