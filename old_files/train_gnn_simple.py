#!/usr/bin/env python3
"""
Simplified GNN Training for SpMV Learning (without edge weights initially)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import load_npz


class SimpleSpMV_GNN(nn.Module):
    """Simplified GNN for SpMV learning without edge weights."""
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers (no edge weights)
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index):
        """Forward pass without edge weights."""
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return x


def load_data():
    """Load and prepare data for training."""
    
    print("Loading dataset...")
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    
    X_train = data['X_train']
    Y_train = data['Y_train'] 
    X_val = data['X_val']
    Y_val = data['Y_val']
    
    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    print(f"Nodes: {X_train.shape[1]}")
    
    # Convert to graph format (connectivity only, no edge weights)
    edge_index, _ = from_scipy_sparse_matrix(K)
    
    print(f"Graph: {edge_index.shape[1]} edges")
    print(f"Input range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Output range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)  # [400, 648]
    Y_train = torch.FloatTensor(Y_train)  # [400, 648]
    X_val = torch.FloatTensor(X_val)      # [100, 648]
    Y_val = torch.FloatTensor(Y_val)      # [100, 648]
    
    return X_train, Y_train, X_val, Y_val, edge_index


def train_epoch(model, X_train, Y_train, edge_index, optimizer, criterion, batch_size=16):
    """Train for one epoch using mini-batches."""
    model.train()
    
    n_samples = X_train.shape[0]
    indices = torch.randperm(n_samples)
    total_loss = 0
    num_batches = 0
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch
        x_batch = X_train[batch_indices]  # [batch_size, 648]
        y_batch = Y_train[batch_indices]  # [batch_size, 648]
        
        batch_loss = 0
        optimizer.zero_grad()
        
        # Process each sample in batch (since graph structure is the same)
        for i in range(x_batch.shape[0]):
            x_sample = x_batch[i].unsqueeze(-1)  # [648, 1]
            y_sample = y_batch[i]                # [648]
            
            # Forward pass
            pred = model(x_sample, edge_index).squeeze(-1)  # [648]
            
            # Compute loss
            loss = criterion(pred, y_sample)
            batch_loss += loss
        
        # Average loss over batch
        batch_loss = batch_loss / x_batch.shape[0]
        
        # Backward pass
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, X_val, Y_val, edge_index, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x_sample = X_val[i].unsqueeze(-1)  # [648, 1]
            y_sample = Y_val[i]                # [648]
            
            pred = model(x_sample, edge_index).squeeze(-1)
            loss = criterion(pred, y_sample)
            total_loss += loss.item()
    
    return total_loss / X_val.shape[0]


def plot_training_results(train_losses, val_losses, save_dir):
    """Plot training results."""
    
    plt.figure(figsize=(15, 5))
    
    # Loss curves (log scale)
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss curves (linear scale)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final losses comparison
    plt.subplot(1, 3, 3)
    final_train = train_losses[-10:]  # Last 10 epochs
    final_val = val_losses[-10:]
    final_epochs = range(len(train_losses)-9, len(train_losses)+1)
    plt.plot(final_epochs, final_train, 'b-', label='Training', linewidth=2)
    plt.plot(final_epochs, final_val, 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Final Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def test_inference(model, X_val, Y_val, edge_index, save_dir):
    """Test inference on validation set."""
    
    model.eval()
    
    print("\n=== INFERENCE TESTING ===")
    
    results = []
    
    with torch.no_grad():
        for i in range(min(5, X_val.shape[0])):
            x_sample = X_val[i].unsqueeze(-1)  # [648, 1]
            y_true = Y_val[i]                  # [648]
            
            pred = model(x_sample, edge_index).squeeze(-1)  # [648]
            
            mse = F.mse_loss(pred, y_true).item()
            mae = F.l1_loss(pred, y_true).item()
            
            results.append({
                'sample': i,
                'mse': mse,
                'mae': mae,
                'x_input': x_sample.squeeze().numpy(),
                'y_true': y_true.numpy(),
                'y_pred': pred.numpy()
            })
            
            print(f"Sample {i}: MSE = {mse:.6f}, MAE = {mae:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 3 * len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Input
        axes[i, 0].plot(result['x_input'], 'b-', linewidth=1)
        axes[i, 0].set_title(f'Sample {i}: Input f')
        axes[i, 0].grid(True, alpha=0.3)
        
        # True vs Predicted
        axes[i, 1].plot(result['y_true'], 'r-', linewidth=1, label='True')
        axes[i, 1].plot(result['y_pred'], 'g--', linewidth=1, label='Predicted')
        axes[i, 1].set_title(f'Output (MSE={result["mse"]:.2e})')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Error
        error = np.abs(result['y_true'] - result['y_pred'])
        axes[i, 2].plot(error, 'k-', linewidth=1)
        axes[i, 2].set_title(f'|Error| (MAE={result["mae"]:.2e})')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'inference_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Inference results saved to: {save_path}")
    plt.close()
    
    return results


def main():
    # Load data
    X_train, Y_train, X_val, Y_val, edge_index = load_data()
    
    # Create model
    model = SimpleSpMV_GNN(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.1)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {num_params:,} parameters")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gnn_simple_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    epochs = 100
    print(f"\nTraining for {epochs} epochs...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, X_train, Y_train, edge_index, optimizer, criterion, batch_size=8)
        
        # Validate
        val_loss = evaluate(model, X_val, Y_val, edge_index, criterion)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs}: Train = {train_loss:.6f}, Val = {val_loss:.6f}")
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    
    # Plot results
    plot_training_results(train_losses, val_losses, save_dir)
    
    # Save training history
    np.savez(os.path.join(save_dir, 'training_history.npz'),
             train_losses=train_losses,
             val_losses=val_losses,
             best_val_loss=best_val_loss)
    
    # Load best model and test inference
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    inference_results = test_inference(model, X_val, Y_val, edge_index, save_dir)
    
    print(f"\nAll results saved to: {save_dir}")
    
    return model, inference_results


if __name__ == "__main__":
    model, results = main()
