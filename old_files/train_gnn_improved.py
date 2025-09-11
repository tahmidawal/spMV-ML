#!/usr/bin/env python3
"""
Improved GNN Training for SpMV Learning
- Better edge weight normalization
- Residual connections
- Layer normalization
- Graph attention
- Deeper architecture
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


class ImprovedSpMV_GNN(nn.Module):
    """
    Improved GNN for SpMV learning with:
    - Residual connections
    - Layer normalization  
    - Better edge weight handling
    - Attention mechanism option
    """
    
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, num_layers=4, 
                 conv_type='GCN', dropout=0.1, use_residual=True, use_layer_norm=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.conv_type = conv_type
        
        # Input projection (to match hidden_dim for residuals)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if use_residual else None
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First layer
        first_input_dim = input_dim if not use_residual else hidden_dim
        if conv_type == 'GCN':
            self.convs.append(GCNConv(first_input_dim, hidden_dim))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(first_input_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(first_input_dim, hidden_dim))
        
        if use_layer_norm:
            self.layer_norms.append(LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
            elif conv_type == 'SAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            if use_layer_norm:
                self.layer_norms.append(LayerNorm(hidden_dim))
        
        # Output layer
        self.output_conv = GCNConv(hidden_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim) if conv_type != 'GCN' else None
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass with residual connections and layer normalization."""
        
        # Input projection for residual connections
        if self.use_residual and self.input_proj is not None:
            x = self.input_proj(x)
            residual = x
        
        # First layer
        if self.conv_type == 'GCN' and edge_weight is not None:
            x = self.convs[0](x, edge_index, edge_weight)
        else:
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
            
            if self.conv_type == 'GCN' and edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.use_residual:
                x = x + residual
        
        # Output layer (no activation, no residual)
        if self.conv_type == 'GCN' and edge_weight is not None:
            x = self.output_conv(x, edge_index, edge_weight)
        else:
            x = self.output_conv(x, edge_index)
        
        return x


def normalize_edge_weights_robust(edge_weight):
    """
    Robust edge weight normalization to prevent gradient explosion.
    Uses log-transform for large values.
    """
    
    # Method: Sign-preserving log transform
    sign = torch.sign(edge_weight)
    abs_weight = torch.abs(edge_weight)
    
    # Log transform: sign(w) * log(1 + |w|)
    log_weight = sign * torch.log(1 + abs_weight)
    
    # Then standard normalization
    log_weight_std = log_weight.std()
    if log_weight_std > 0:
        normalized_weight = log_weight / log_weight_std
    else:
        normalized_weight = log_weight
    
    return normalized_weight, log_weight_std


def load_data_improved():
    """Load data with improved edge weight handling."""
    
    print("Loading dataset...")
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    
    X_train = data['X_train']
    Y_train = data['Y_train'] 
    X_val = data['X_val']
    Y_val = data['Y_val']
    
    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    
    # Convert to graph format
    edge_index, edge_weight = from_scipy_sparse_matrix(K)
    edge_weight = edge_weight.float()
    
    print(f"Original edge weight range: [{edge_weight.min():.3f}, {edge_weight.max():.3f}]")
    
    # Apply robust normalization
    edge_weight_norm, norm_factor = normalize_edge_weights_robust(edge_weight)
    print(f"Normalized edge weight range: [{edge_weight_norm.min():.3f}, {edge_weight_norm.max():.3f}]")
    print(f"Normalization factor: {norm_factor:.3f}")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)
    X_val = torch.FloatTensor(X_val)
    Y_val = torch.FloatTensor(Y_val)
    
    return X_train, Y_train, X_val, Y_val, edge_index, edge_weight_norm


def train_epoch_improved(model, X_train, Y_train, edge_index, edge_weight, optimizer, criterion, batch_size=8):
    """Improved training with better batch handling."""
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
            
            # Forward pass
            pred = model(x_sample, edge_index, edge_weight).squeeze(-1)
            
            # Compute loss
            loss = criterion(pred, y_sample)
            batch_loss += loss
        
        # Average loss over batch
        batch_loss = batch_loss / x_batch.shape[0]
        
        # Backward pass
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Stricter clipping
        
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_improved(model, X_val, Y_val, edge_index, edge_weight, criterion):
    """Improved evaluation."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x_sample = X_val[i].unsqueeze(-1)
            y_sample = Y_val[i]
            
            pred = model(x_sample, edge_index, edge_weight).squeeze(-1)
            loss = criterion(pred, y_sample)
            total_loss += loss.item()
    
    return total_loss / X_val.shape[0]


def plot_improved_results(train_losses, val_losses, save_dir):
    """Plot improved training results."""
    
    plt.figure(figsize=(18, 6))
    
    # Loss curves (log scale)
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss curves (linear scale)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate and convergence analysis
    plt.subplot(1, 3, 3)
    # Plot loss improvement rate
    if len(train_losses) > 10:
        train_smooth = np.convolve(train_losses, np.ones(10)/10, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(10)/10, mode='valid')
        smooth_epochs = range(10, len(train_losses) + 1)
        plt.plot(smooth_epochs, train_smooth, 'b-', label='Train (smoothed)', linewidth=2)
        plt.plot(smooth_epochs, val_smooth, 'r-', label='Val (smoothed)', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Smoothed Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'improved_training_curves.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Improved training curves saved to: {save_path}")
    plt.close()


def detailed_inference_analysis(model, X_val, Y_val, edge_index, edge_weight, save_dir):
    """Detailed inference analysis with accuracy metrics."""
    
    model.eval()
    print("\n=== DETAILED INFERENCE ANALYSIS ===")
    
    all_mse = []
    all_mae = []
    all_relative_errors = []
    
    # Test on all validation samples
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x_sample = X_val[i].unsqueeze(-1)
            y_true = Y_val[i]
            
            pred = model(x_sample, edge_index, edge_weight).squeeze(-1)
            
            mse = F.mse_loss(pred, y_true).item()
            mae = F.l1_loss(pred, y_true).item()
            
            # Relative error
            y_range = y_true.max() - y_true.min()
            relative_error = mae / y_range.item() if y_range > 0 else 0
            
            all_mse.append(mse)
            all_mae.append(mae)
            all_relative_errors.append(relative_error)
    
    # Statistics
    print(f"Validation Statistics (all {len(all_mse)} samples):")
    print(f"  MSE: mean={np.mean(all_mse):.6f}, std={np.std(all_mse):.6f}")
    print(f"  MAE: mean={np.mean(all_mae):.6f}, std={np.std(all_mae):.6f}")
    print(f"  Relative Error: mean={np.mean(all_relative_errors):.4f}, std={np.std(all_relative_errors):.4f}")
    print(f"  Best sample MSE: {np.min(all_mse):.6f}")
    print(f"  Worst sample MSE: {np.max(all_mse):.6f}")
    
    # Plot detailed inference results for best and worst cases
    best_idx = np.argmin(all_mse)
    worst_idx = np.argmax(all_mse)
    sample_indices = [best_idx, worst_idx, 0, 1, 2]  # Best, worst, and first 3
    
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(20, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(sample_indices):
            x_sample = X_val[sample_idx].unsqueeze(-1)
            y_true = Y_val[sample_idx]
            pred = model(x_sample, edge_index, edge_weight).squeeze(-1)
            
            mse = all_mse[sample_idx]
            mae = all_mae[sample_idx]
            rel_err = all_relative_errors[sample_idx]
            
            # Convert to numpy
            x_np = x_sample.squeeze().numpy()
            y_true_np = y_true.numpy()
            y_pred_np = pred.numpy()
            error_np = np.abs(y_true_np - y_pred_np)
            
            # Plot 1: Input field
            axes[plot_idx, 0].plot(x_np, 'b-', linewidth=1, alpha=0.8)
            axes[plot_idx, 0].set_title(f'Sample {sample_idx}: Input f')
            axes[plot_idx, 0].grid(True, alpha=0.3)
            
            # Plot 2: True vs Predicted
            axes[plot_idx, 1].plot(y_true_np, 'r-', linewidth=1.5, alpha=0.8, label='True')
            axes[plot_idx, 1].plot(y_pred_np, 'g--', linewidth=1.5, alpha=0.8, label='Predicted')
            axes[plot_idx, 1].set_title(f'Output (MSE={mse:.2e}, MAE={mae:.2e})')
            axes[plot_idx, 1].legend()
            axes[plot_idx, 1].grid(True, alpha=0.3)
            
            # Plot 3: Error
            axes[plot_idx, 2].plot(error_np, 'k-', linewidth=1, alpha=0.8)
            axes[plot_idx, 2].set_title(f'|Error| (Rel={rel_err:.3f})')
            axes[plot_idx, 2].grid(True, alpha=0.3)
            axes[plot_idx, 2].set_yscale('log')
            
            # Plot 4: Scatter plot (true vs predicted)
            axes[plot_idx, 3].scatter(y_true_np, y_pred_np, alpha=0.6, s=8)
            
            # Perfect prediction line
            y_min, y_max = min(y_true_np.min(), y_pred_np.min()), max(y_true_np.max(), y_pred_np.max())
            axes[plot_idx, 3].plot([y_min, y_max], [y_min, y_max], 'r--', alpha=0.8)
            
            axes[plot_idx, 3].set_xlabel('True')
            axes[plot_idx, 3].set_ylabel('Predicted')
            axes[plot_idx, 3].set_title(f'True vs Pred (R²≈{1-mse/np.var(y_true_np):.3f})')
            axes[plot_idx, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'detailed_inference_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Detailed inference analysis saved to: {save_path}")
    plt.close()
    
    return {
        'mean_mse': np.mean(all_mse),
        'mean_mae': np.mean(all_mae),
        'mean_relative_error': np.mean(all_relative_errors),
        'best_mse': np.min(all_mse),
        'worst_mse': np.max(all_mse)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Improved GNN for SpMV learning')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--conv-type', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE'])
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use-edge-weights', action='store_true', help='Use edge weights')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    args = parser.parse_args()
    
    # Load data
    X_train, Y_train, X_val, Y_val, edge_index, edge_weight = load_data_improved()
    
    # Create model
    model = ImprovedSpMV_GNN(
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
    print(f"Improved model: {num_params:,} parameters")
    print(f"Architecture: {args.conv_type} with {args.num_layers} layers, hidden_dim={args.hidden_dim}")
    print(f"Features: Residual connections, Layer normalization, Edge weights={args.use_edge_weights}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gnn_improved_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting improved training for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    edge_weights_to_use = edge_weight if args.use_edge_weights else None
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch_improved(model, X_train, Y_train, edge_index, edge_weights_to_use, 
                                        optimizer, criterion, args.batch_size)
        
        # Validate
        val_loss = evaluate_improved(model, X_val, Y_val, edge_index, edge_weights_to_use, criterion)
        
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
                'args': vars(args)
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train = {train_loss:.6f}, Val = {val_loss:.6f}, LR = {current_lr:.2e}")
    
    print(f"\nImproved training completed! Best validation loss: {best_val_loss:.6f}")
    
    # Load best model for final analysis
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot results
    plot_improved_results(train_losses, val_losses, save_dir)
    
    # Detailed inference analysis
    inference_stats = detailed_inference_analysis(model, X_val, Y_val, edge_index, edge_weights_to_use, save_dir)
    
    # Save comprehensive results
    results_summary = {
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'inference_stats': inference_stats,
        'model_params': num_params,
        'args': vars(args)
    }
    
    with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nIMPROVED MODEL SUMMARY:")
    print(f"  Best MSE: {best_val_loss:.6f} (vs previous: 0.013632)")
    print(f"  Improvement: {(0.013632 - best_val_loss)/0.013632*100:.1f}%")
    print(f"  Mean relative error: {inference_stats['mean_relative_error']:.4f}")
    
    return model, results_summary


if __name__ == "__main__":
    model, results = main()
