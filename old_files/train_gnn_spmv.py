#!/usr/bin/env python3
"""
Graph Neural Network for Learning Sparse Matrix-Vector Multiplication (SpMV)

This script trains a GNN to learn the forward operation: u = K @ f
where K is a sparse FEM stiffness matrix and f is a sinusoidal field.
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

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader  # Updated import
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.utils import from_scipy_sparse_matrix
    print(f"Using PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    print("ERROR: PyTorch Geometric not installed!")
    print("Install with: pip install torch-geometric")
    exit(1)

from scipy.sparse import load_npz


class SpMV_GNN(nn.Module):
    """
    Graph Neural Network for learning Sparse Matrix-Vector Multiplication.
    
    Architecture:
    - Input: Node features (field values f[i])
    - Graph: Mesh connectivity from sparse matrix K
    - Output: Predicted SpMV result u = K @ f
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, 
                 conv_type='GCN', dropout=0.1):
        super(SpMV_GNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if conv_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
            
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif conv_type == 'SAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if conv_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
        
        Returns:
            Node predictions [num_nodes, output_dim]
        """
        
        for i in range(self.num_layers - 1):
            # Graph convolution
            if self.conv_type == 'GCN' and edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation, no dropout)
        if self.conv_type == 'GCN' and edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight)
        else:
            x = self.convs[-1](x, edge_index)
        
        return x


def load_dataset(dataset_path, sparse_matrix_path):
    """Load the forward SpMV dataset and convert to graph format."""
    
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    K = load_npz(sparse_matrix_path)
    
    # Extract data
    X_train = data['X_train']  # [400, 648]
    Y_train = data['Y_train']  # [400, 648]
    X_val = data['X_val']      # [100, 648]
    Y_val = data['Y_val']      # [100, 648]
    
    print(f"Dataset loaded: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    print(f"Node dimension: {X_train.shape[1]} nodes")
    print(f"Sparse matrix: {K.shape}, {K.nnz} non-zeros")
    
    # Convert sparse matrix to PyTorch Geometric format
    edge_index, edge_weight = from_scipy_sparse_matrix(K)
    edge_weight = edge_weight.float()  # Ensure float32 dtype
    
    # Normalize edge weights to prevent gradient explosion
    edge_weight_std = edge_weight.std()
    edge_weight = edge_weight / edge_weight_std
    print(f"Edge weights normalized by std = {edge_weight_std:.3f}")
    print(f"Normalized edge weight range: [{edge_weight.min():.3f}, {edge_weight.max():.3f}]")
    
    print(f"Graph: {edge_index.shape[1]} edges")
    
    # Convert to PyTorch tensors and normalize Y by same factor
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # [400, 648, 1]
    Y_train = torch.FloatTensor(Y_train / edge_weight_std.numpy()).unsqueeze(-1)  # [400, 648, 1] - normalized
    X_val = torch.FloatTensor(X_val).unsqueeze(-1)      # [100, 648, 1]
    Y_val = torch.FloatTensor(Y_val / edge_weight_std.numpy()).unsqueeze(-1)      # [100, 648, 1] - normalized
    
    print(f"Target values also normalized by same factor for consistency")
    
    return {
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'edge_index': edge_index, 'edge_weight': edge_weight,
        'edge_weight_std': edge_weight_std,  # Store for denormalization
        'metadata': data
    }


def create_data_loaders(dataset, batch_size=32):
    """Create PyTorch Geometric data loaders."""
    
    # For graph-level tasks, we create Data objects for each sample
    train_data_list = []
    val_data_list = []
    
    edge_index = dataset['edge_index']
    edge_weight = dataset['edge_weight']
    
    # Training data
    for i in range(dataset['X_train'].shape[0]):
        data = Data(
            x=dataset['X_train'][i],  # [648, 1]
            y=dataset['Y_train'][i],  # [648, 1]
            edge_index=edge_index,
            edge_attr=edge_weight
        )
        train_data_list.append(data)
    
    # Validation data
    for i in range(dataset['X_val'].shape[0]):
        data = Data(
            x=dataset['X_val'][i],   # [648, 1]
            y=dataset['Y_val'][i],   # [648, 1]
            edge_index=edge_index,
            edge_attr=edge_weight
        )
        val_data_list.append(data)
    
    # Create data loaders
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        
        # Compute loss
        loss = criterion(pred, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
    
    return total_loss / num_samples


def evaluate(model, val_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Compute loss
            loss = criterion(pred, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs
    
    return total_loss / num_samples


def plot_loss_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Loss curves (linear scale)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Loss curves saved to: {save_path}")
        plt.close()
    # plt.show()  # Disabled to prevent hanging


def inference_analysis(model, dataset, device, save_dir=None):
    """Perform detailed inference analysis."""
    
    model.eval()
    
    print("\n=== INFERENCE ANALYSIS ===")
    
    # Test on a few validation samples
    edge_index = dataset['edge_index'].to(device)
    edge_weight = dataset['edge_weight'].to(device)
    
    test_indices = [0, 1, 2, 3, 4]  # First 5 validation samples
    
    results = []
    
    with torch.no_grad():
        for i in test_indices:
            # Get sample
            x_input = dataset['X_val'][i:i+1].to(device)  # [1, 648, 1]
            y_true = dataset['Y_val'][i:i+1].to(device)   # [1, 648, 1]
            
            # Predict
            y_pred = model(x_input.squeeze(0), edge_index, edge_weight)  # [648, 1]
            y_pred = y_pred.unsqueeze(0)  # [1, 648, 1]
            
            # Compute metrics
            mse = F.mse_loss(y_pred, y_true).item()
            mae = F.l1_loss(y_pred, y_true).item()
            
            # Convert to numpy for analysis
            x_np = x_input.cpu().numpy().squeeze()  # [648]
            y_true_np = y_true.cpu().numpy().squeeze()  # [648]
            y_pred_np = y_pred.cpu().numpy().squeeze()  # [648]
            
            results.append({
                'sample': i,
                'mse': mse,
                'mae': mae,
                'x_input': x_np,
                'y_true': y_true_np,
                'y_pred': y_pred_np
            })
            
            print(f"Sample {i}: MSE = {mse:.6f}, MAE = {mae:.6f}")
    
    # Plot inference results
    if save_dir:
        plot_inference_results(results, save_dir)
    
    return results


def plot_inference_results(results, save_dir):
    """Plot inference results comparison."""
    
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        x_input = result['x_input']
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        # Plot 1: Input
        axes[i, 0].plot(x_input, 'b-', linewidth=1, alpha=0.8)
        axes[i, 0].set_title(f'Sample {result["sample"]}: Input f')
        axes[i, 0].set_xlabel('Node index')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2: True vs Predicted
        axes[i, 1].plot(y_true, 'r-', linewidth=1, alpha=0.8, label='True')
        axes[i, 1].plot(y_pred, 'g--', linewidth=1, alpha=0.8, label='Predicted')
        axes[i, 1].set_title(f'Output Comparison (MSE={result["mse"]:.2e})')
        axes[i, 1].set_xlabel('Node index')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error
        error = np.abs(y_true - y_pred)
        axes[i, 2].plot(error, 'k-', linewidth=1, alpha=0.8)
        axes[i, 2].set_title(f'Absolute Error (MAE={result["mae"]:.2e})')
        axes[i, 2].set_xlabel('Node index')
        axes[i, 2].set_ylabel('|True - Predicted|')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_yscale('log')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'inference_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Inference results saved to: {save_path}")
    plt.close()
    # plt.show()  # Disabled to prevent hanging


def main():
    parser = argparse.ArgumentParser(description='Train GNN for SpMV learning')
    parser.add_argument('--dataset', type=str, 
                       default='ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz',
                       help='Path to dataset NPZ file')
    parser.add_argument('--sparse-matrix', type=str,
                       default='ml_data/sparse_matrix_K_20250910_161619.npz',
                       help='Path to sparse matrix NPZ file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--conv-type', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE'],
                       help='Graph convolution type')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save-dir', type=str, default='gnn_results', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.sparse_matrix)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset, args.batch_size)
    
    print(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    model = SpMV_GNN(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {num_params:,} trainable parameters")
    print(f"Architecture: {args.conv_type} with {args.num_layers} layers, hidden_dim={args.hidden_dim}")
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        
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
        if (epoch + 1) % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    
    # Plot loss curves
    loss_plot_path = os.path.join(save_dir, 'loss_curves.png')
    plot_loss_curves(train_losses, val_losses, loss_plot_path)
    
    # Save training history
    np.savez(os.path.join(save_dir, 'training_history.npz'),
             train_losses=train_losses,
             val_losses=val_losses,
             best_val_loss=best_val_loss)
    
    # Load best model for inference
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    # Perform inference analysis
    inference_results = inference_analysis(model, dataset, device, save_dir)
    
    print(f"\nAll results saved to: {save_dir}")
    
    return model, dataset, inference_results


if __name__ == "__main__":
    model, dataset, results = main()
