#!/usr/bin/env python3
"""
Best GNN Training for SpMV Learning
Based on the successful improved architecture but without edge weights to avoid NaN issues.
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LayerNorm
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import load_npz


class BestSpMV_GNN(nn.Module):
    """
    Best performing GNN architecture for SpMV learning.
    Uses the successful architecture without edge weights.
    """
    
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, num_layers=4, 
                 conv_type='GCN', dropout=0.1, use_residual=True, use_layer_norm=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.conv_type = conv_type
        
        # Input projection for residual connections
        self.input_proj = nn.Linear(input_dim, hidden_dim) if use_residual else None
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First layer
        first_input_dim = input_dim if not use_residual else hidden_dim
        if conv_type == 'GCN':
            self.convs.append(GCNConv(first_input_dim, hidden_dim))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(first_input_dim, hidden_dim, heads=8, concat=False, dropout=dropout))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(first_input_dim, hidden_dim))
        
        if use_layer_norm:
            self.layer_norms.append(LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=False, dropout=dropout))
            elif conv_type == 'SAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            if use_layer_norm:
                self.layer_norms.append(LayerNorm(hidden_dim))
        
        # Output layer
        self.output_conv = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        """Forward pass with residual connections and layer normalization."""
        
        # Input projection for residual connections
        if self.use_residual and self.input_proj is not None:
            x = self.input_proj(x)
            residual = x
        
        # First layer
        x = self.convs[0](x, edge_index)
        
        if self.use_layer_norm:
            x = self.layer_norms[0](x)
        
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Add residual connection
        if self.use_residual and self.input_proj is not None:
            x = x + residual
        
        # Hidden layers with residual connections
        for i in range(1, self.num_layers - 1):
            residual = x
            
            x = self.convs[i](x, edge_index)
            
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.use_residual:
                x = x + residual
        
        # Output layer (no activation, no residual)
        x = self.output_conv(x, edge_index)
        
        return x


def load_data_best():
    """Load data for best model (without edge weight issues)."""
    
    print("Loading dataset...")
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    
    X_train = data['X_train']
    Y_train = data['Y_train'] 
    X_val = data['X_val']
    Y_val = data['Y_val']
    
    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    print(f"Input range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Output range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")
    
    # Convert to graph format (connectivity only - no edge weights to avoid NaN)
    edge_index, _ = from_scipy_sparse_matrix(K)
    print(f"Graph: {edge_index.shape[1]} edges (topology only)")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val)
    Y_val = torch.FloatTensor(Y_val)
    
    return X_train, Y_train, X_val, Y_val, edge_index


def train_epoch_best(model, X_train, Y_train, edge_index, optimizer, criterion, batch_size=8):
    """Best training approach without edge weights."""
    model.train()
    
    n_samples = X_train.shape[0]
    indices = torch.randperm(n_samples)
    total_loss = 0
    num_batches = 0
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        x_batch = X_train[batch_indices]
        y_batch = Y_train[batch_indices]
        
        optimizer.zero_grad()
        batch_loss = 0
        
        # Process each sample in batch
        for i in range(x_batch.shape[0]):
            x_sample = x_batch[i].unsqueeze(-1)  # [648, 1]
            y_sample = y_batch[i]                # [648]
            
            # Forward pass (no edge weights)
            pred = model(x_sample, edge_index).squeeze(-1)
            
            # Compute loss
            loss = criterion(pred, y_sample)
            batch_loss += loss
        
        # Average loss over batch
        batch_loss = batch_loss / x_batch.shape[0]
        
        # Backward pass
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_best(model, X_val, Y_val, edge_index, criterion):
    """Evaluate without edge weights."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x_sample = X_val[i].unsqueeze(-1)
            y_sample = Y_val[i]
            
            pred = model(x_sample, edge_index).squeeze(-1)
            loss = criterion(pred, y_sample)
            total_loss += loss.item()
    
    return total_loss / X_val.shape[0]


def plot_training_comparison(train_losses, val_losses, save_dir, comparison_mse=0.013632):
    """Plot training results with comparison to previous model."""
    
    plt.figure(figsize=(18, 6))
    
    # Loss curves (log scale)
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2, alpha=0.8)
    
    # Add comparison line
    plt.axhline(y=comparison_mse, color='orange', linestyle='--', alpha=0.8, 
                label=f'Previous Model ({comparison_mse:.6f})')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Final performance comparison
    plt.subplot(1, 3, 2)
    models = ['Previous\nSimple', 'Current\nImproved']
    mse_values = [comparison_mse, min(val_losses)]
    improvement = (comparison_mse - min(val_losses)) / comparison_mse * 100
    
    bars = plt.bar(models, mse_values, color=['orange', 'green'], alpha=0.7)
    plt.ylabel('Best Validation MSE')
    plt.title(f'Model Comparison\n{improvement:.1f}% Improvement')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{val:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # Training convergence
    plt.subplot(1, 3, 3)
    if len(train_losses) > 20:
        final_epochs = range(len(train_losses)-19, len(train_losses)+1)
        final_train = train_losses[-20:]
        final_val = val_losses[-20:]
        plt.plot(final_epochs, final_train, 'b-', label='Training', linewidth=2)
        plt.plot(final_epochs, final_val, 'r-', label='Validation', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Final Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'best_training_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Training comparison saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Best GNN for SpMV learning')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--conv-type', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE'])
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    args = parser.parse_args()
    
    # Load data (without edge weights to avoid NaN)
    X_train, Y_train, X_val, Y_val, edge_index = load_data_best()
    
    # Create improved model
    model = BestSpMV_GNN(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        dropout=args.dropout,
        use_residual=True,
        use_layer_norm=True
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Best model: {num_params:,} parameters")
    print(f"Architecture: {args.conv_type} with {args.num_layers} layers, hidden_dim={args.hidden_dim}")
    print(f"Features: Residual connections, Layer normalization (NO edge weights to avoid NaN)")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gnn_best_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Training loop
    print(f"\nStarting best model training for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch_best(model, X_train, Y_train, edge_index, optimizer, criterion, args.batch_size)
        
        # Validate  
        val_loss = evaluate_best(model, X_val, Y_val, edge_index, criterion)
        
        # Update scheduler
        scheduler.step()
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': vars(args)
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train = {train_loss:.6f}, Val = {val_loss:.6f}, LR = {current_lr:.2e}")
    
    print(f"\nBest model training completed! Best validation loss: {best_val_loss:.6f}")
    
    # Load best model for analysis
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training results with comparison
    plot_training_comparison(train_losses, val_losses, save_dir, comparison_mse=0.013632)
    
    # Detailed inference analysis
    model.eval()
    all_mse = []
    all_mae = []
    
    print("\n=== FINAL ACCURACY ANALYSIS ===")
    
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x_sample = X_val[i].unsqueeze(-1)
            y_true = Y_val[i]
            pred = model(x_sample, edge_index).squeeze(-1)
            
            mse = F.mse_loss(pred, y_true).item()
            mae = F.l1_loss(pred, y_true).item()
            
            all_mse.append(mse)
            all_mae.append(mae)
    
    # Final statistics
    mean_mse = np.mean(all_mse)
    mean_mae = np.mean(all_mae)
    
    print(f"Final Model Performance:")
    print(f"  Best validation MSE: {best_val_loss:.6f}")
    print(f"  Mean MSE: {mean_mse:.6f}")
    print(f"  Mean MAE: {mean_mae:.6f}")
    print(f"  RMSE: {np.sqrt(mean_mse):.6f}")
    print(f"  Relative error: ~{mean_mae/1.2*100:.2f}% (vs output range ~1.2)")
    
    # Comparison with previous models
    print(f"\nAccuracy Comparison:")
    print(f"  Previous simple model MSE: 0.013632")
    print(f"  Current best model MSE: {best_val_loss:.6f}")
    improvement = (0.013632 - best_val_loss) / 0.013632 * 100
    print(f"  Improvement: {improvement:.1f}% better! 🎉")
    
    # Save final results
    results_summary = {
        'best_val_loss': best_val_loss,
        'mean_mse': mean_mse,
        'mean_mae': mean_mae,
        'improvement_percent': improvement,
        'model_params': num_params,
        'args': vars(args)
    }
    
    with open(os.path.join(save_dir, 'best_results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save training history
    np.savez(os.path.join(save_dir, 'best_training_history.npz'),
             train_losses=train_losses,
             val_losses=val_losses,
             best_val_loss=best_val_loss)
    
    print(f"\nAll results saved to: {save_dir}")
    
    return model, results_summary


if __name__ == "__main__":
    model, results = main()

