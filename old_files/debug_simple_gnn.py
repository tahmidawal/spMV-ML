#!/usr/bin/env python3
"""
Simple debug: Try GNN without edge weights first
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

print("=== DEBUGGING GNN WITHOUT EDGE WEIGHTS ===")

# Load data
data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')

X_train = data['X_train'][:5]  # Just 5 samples
Y_train = data['Y_train'][:5]

print(f"Data: {X_train.shape}, {Y_train.shape}")

# Convert to graph (without edge weights)
edge_index, _ = from_scipy_sparse_matrix(K)
print(f"Graph edges: {edge_index.shape[1]}")

# Simple model without edge weights
class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 1)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = SimpleGCN()

# Test without edge weights
x_test = torch.FloatTensor(X_train[0])  # [648]
y_test = torch.FloatTensor(Y_train[0])  # [648]

print(f"Input range: [{x_test.min():.6f}, {x_test.max():.6f}]")
print(f"Target range: [{y_test.min():.6f}, {y_test.max():.6f}]")

print("\nTesting forward pass WITHOUT edge weights...")
try:
    with torch.no_grad():
        pred = model(x_test.unsqueeze(-1), edge_index)  # [648, 1]
        pred = pred.squeeze(-1)  # [648]
        print(f"Prediction range: [{pred.min():.6f}, {pred.max():.6f}]")
        print(f"Has NaN: {torch.isnan(pred).any()}")
        
        if not torch.isnan(pred).any():
            print("✅ SUCCESS: Forward pass works without edge weights!")
            
            # Test training step
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(5):
                optimizer.zero_grad()
                pred = model(x_test.unsqueeze(-1).requires_grad_(True), edge_index).squeeze(-1)
                loss = criterion(pred, y_test)
                
                if torch.isnan(loss):
                    print(f"❌ NaN loss at epoch {epoch}")
                    break
                    
                loss.backward()
                
                # Check gradients
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                
                if any(np.isnan(grad_norms)):
                    print(f"❌ NaN gradients at epoch {epoch}")
                    break
                
                optimizer.step()
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Max grad norm = {max(grad_norms):.6f}")
            
            print("✅ Training works without edge weights!")
        
except Exception as e:
    print(f"❌ Failed: {e}")

print("\nCONCLUSION:")
print("If this works, the issue is with edge weights.")
print("Solution: Train GNN without edge weights initially, or use different edge weight handling.")
