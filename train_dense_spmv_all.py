#!/usr/bin/env python3
"""
Dense Matrix-Matrix SpMV Learning
A novel approach: Convert SpMV to dense MM for GPU efficiency
Instead of y = K @ x (sparse), we learn dense matrices for matrix-matrix multiplication
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class DenseSpMV_Simple(nn.Module):
    """
    Simple dense matrix approach: Learn a single dense matrix W
    Reshape vector to matrix, apply W, reshape back
    """
    def __init__(self, vector_dim=648, matrix_shape=(24, 27)):
        super().__init__()
        self.vector_dim = vector_dim
        self.m, self.n = matrix_shape
        assert self.m * self.n == vector_dim, f"Matrix shape {matrix_shape} doesn't match vector dim {vector_dim}"
        
        # Learn a dense transformation matrix
        self.W = nn.Parameter(torch.randn(self.m, self.m) * 0.02)
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(vector_dim))
        
        self.total_params = self.m * self.m + vector_dim
        
    def forward(self, x):
        """
        x: [batch, vector_dim] or [vector_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Reshape to matrix [batch, m, n]
        X_in = x.view(batch_size, self.m, self.n)
        
        # Dense matrix multiplication [batch, m, m] @ [batch, m, n] = [batch, m, n]
        X_out = torch.bmm(self.W.unsqueeze(0).expand(batch_size, -1, -1), X_in)
        
        # Reshape back to vector
        y = X_out.view(batch_size, self.vector_dim)
        
        # Add bias
        y = y + self.bias
        
        return y.squeeze() if batch_size == 1 else y


class DenseSpMV_TwoMatrix(nn.Module):
    """
    Two-matrix approach: Y = W1 @ X @ W2^T
    More expressive with two learnable matrices
    """
    def __init__(self, vector_dim=648, matrix_shape=(24, 27)):
        super().__init__()
        self.vector_dim = vector_dim
        self.m, self.n = matrix_shape
        assert self.m * self.n == vector_dim
        
        # Two transformation matrices
        self.W1 = nn.Parameter(torch.randn(self.m, self.m) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.n, self.n) * 0.02)
        
        self.bias = nn.Parameter(torch.zeros(vector_dim))
        
        self.total_params = self.m * self.m + self.n * self.n + vector_dim
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Reshape to matrix
        X_in = x.view(batch_size, self.m, self.n)
        
        # Two matrix multiplications: W1 @ X @ W2^T
        X_temp = torch.bmm(self.W1.unsqueeze(0).expand(batch_size, -1, -1), X_in)
        X_out = torch.bmm(X_temp, self.W2.t().unsqueeze(0).expand(batch_size, -1, -1))
        
        # Reshape back
        y = X_out.view(batch_size, self.vector_dim)
        y = y + self.bias
        
        return y.squeeze() if batch_size == 1 else y


class DenseSpMV_LowRank(nn.Module):
    """
    Low-rank factorization: W = U @ V^T
    Reduces parameters while maintaining expressiveness
    """
    def __init__(self, vector_dim=648, matrix_shape=(24, 27), rank=8):
        super().__init__()
        self.vector_dim = vector_dim
        self.m, self.n = matrix_shape
        self.rank = rank
        assert self.m * self.n == vector_dim
        
        # Low-rank factors
        self.U = nn.Parameter(torch.randn(self.m, rank) * 0.1)
        self.V = nn.Parameter(torch.randn(self.m, rank) * 0.1)
        
        self.bias = nn.Parameter(torch.zeros(vector_dim))
        
        self.total_params = 2 * self.m * rank + vector_dim
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Compute W = U @ V^T
        W = torch.mm(self.U, self.V.t())
        
        # Reshape to matrix
        X_in = x.view(batch_size, self.m, self.n)
        
        # Dense multiplication
        X_out = torch.bmm(W.unsqueeze(0).expand(batch_size, -1, -1), X_in)
        
        # Reshape back
        y = X_out.view(batch_size, self.vector_dim)
        y = y + self.bias
        
        return y.squeeze() if batch_size == 1 else y


class DenseSpMV_Adaptive(nn.Module):
    """
    Adaptive approach with learned permutation for optimal reshaping
    """
    def __init__(self, vector_dim=648, matrix_shape=(24, 27), hidden_dim=64):
        super().__init__()
        self.vector_dim = vector_dim
        self.m, self.n = matrix_shape
        assert self.m * self.n == vector_dim
        
        # Learn optimal permutation (soft)
        self.perm_logits = nn.Parameter(torch.randn(vector_dim, vector_dim) * 0.01)
        
        # Main transformation
        self.W = nn.Parameter(torch.randn(self.m, self.m) * 0.02)
        
        # Output projection
        self.output_proj = nn.Linear(vector_dim, vector_dim)
        
        self.total_params = vector_dim * vector_dim + self.m * self.m + vector_dim * vector_dim
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Apply soft permutation
        perm_matrix = F.softmax(self.perm_logits, dim=1)
        x_perm = torch.mm(x, perm_matrix.t())
        
        # Reshape to matrix
        X_in = x_perm.view(batch_size, self.m, self.n)
        
        # Dense multiplication
        X_out = torch.bmm(self.W.unsqueeze(0).expand(batch_size, -1, -1), X_in)
        
        # Reshape back
        y = X_out.view(batch_size, self.vector_dim)
        
        # Output projection
        y = self.output_proj(y)
        
        return y.squeeze() if batch_size == 1 else y


def find_optimal_shapes(dim=648):
    """Find all valid matrix shapes for reshaping."""
    shapes = []
    for m in range(2, int(np.sqrt(dim)) + 1):
        if dim % m == 0:
            n = dim // m
            shapes.append((m, n))
    return shapes


def load_data():
    """Load the SpMV dataset."""
    print("Loading dataset...")
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    
    X_train = torch.FloatTensor(data['X_train'])
    Y_train = torch.FloatTensor(data['Y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    Y_val = torch.FloatTensor(data['Y_val'])
    
    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    print(f"Vector dimension: {X_train.shape[1]}")
    
    return X_train, Y_train, X_val, Y_val


def train_epoch(model, X_train, Y_train, optimizer, criterion, batch_size=32, device='cpu'):
    """Train for one epoch."""
    model.train()
    n_samples = X_train.shape[0]
    indices = torch.randperm(n_samples)
    total_loss = 0
    num_batches = 0
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        x_batch = X_train[batch_indices].to(device)
        y_batch = Y_train[batch_indices].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, X_val, Y_val, criterion, device='cpu'):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        # Process in batches for memory efficiency
        batch_size = 50
        n_samples = X_val.shape[0]
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            x_batch = X_val[start_idx:end_idx].to(device)
            y_batch = Y_val[start_idx:end_idx].to(device)
            
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            
            total_loss += loss.item() * (end_idx - start_idx)
    
    return total_loss / n_samples


def benchmark_inference_speed(model, X_val, device='cpu', num_runs=100):
    """Benchmark inference speed."""
    model.eval()
    x_sample = X_val[:10].to(device)  # Use 10 samples
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_sample)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x_sample)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = 10 / avg_time  # samples per second
    
    return avg_time * 1000, throughput  # ms, samples/sec


def main():
    parser = argparse.ArgumentParser(description='Train Dense Matrix SpMV models')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'two_matrix', 'low_rank', 'adaptive', 'all'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--matrix-shape', type=str, default='auto',
                       help='Matrix shape as "m,n" or "auto" to try multiple')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, Y_train, X_val, Y_val = load_data()
    
    # Find valid matrix shapes
    valid_shapes = find_optimal_shapes(648)
    print(f"\nValid matrix shapes for 648-dim vector: {valid_shapes}")
    
    # Determine shapes to try
    if args.matrix_shape == 'auto':
        shapes_to_try = valid_shapes[:3]  # Try first 3 shapes
    else:
        m, n = map(int, args.matrix_shape.split(','))
        shapes_to_try = [(m, n)]
    
    # Determine models to train
    if args.model == 'all':
        model_types = ['simple', 'two_matrix', 'low_rank']
    else:
        model_types = [args.model]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"dense_spmv_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("\n" + "="*80)
    print("TRAINING DENSE MATRIX MODELS FOR SPMV")
    print("="*80)
    
    for model_type in model_types:
        for shape in shapes_to_try:
            print(f"\n{'='*60}")
            print(f"Training {model_type} model with shape {shape}")
            print(f"{'='*60}")
            
            # Create model
            if model_type == 'simple':
                model = DenseSpMV_Simple(vector_dim=648, matrix_shape=shape)
            elif model_type == 'two_matrix':
                model = DenseSpMV_TwoMatrix(vector_dim=648, matrix_shape=shape)
            elif model_type == 'low_rank':
                model = DenseSpMV_LowRank(vector_dim=648, matrix_shape=shape, rank=16)
            elif model_type == 'adaptive':
                model = DenseSpMV_Adaptive(vector_dim=648, matrix_shape=shape)
            
            model = model.to(device)
            
            print(f"Model parameters: {model.total_params:,}")
            print(f"GCN baseline parameters: 50,689")
            print(f"Parameter reduction: {(1 - model.total_params/50689)*100:.1f}%")
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            for epoch in range(args.epochs):
                train_loss = train_epoch(model, X_train, Y_train, optimizer, criterion, 
                                       args.batch_size, device)
                val_loss = evaluate(model, X_val, Y_val, criterion, device)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_type': model_type,
                        'matrix_shape': shape,
                        'val_loss': val_loss,
                        'epoch': epoch
                    }, os.path.join(save_dir, f'{model_type}_{shape[0]}x{shape[1]}_best.pth'))
                
                scheduler.step()
                
                if (epoch + 1) % 30 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}: "
                          f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                          f"Best={best_val_loss:.6f}")
            
            # Benchmark inference speed
            inference_time, throughput = benchmark_inference_speed(model, X_val, device)
            
            # Store results
            result_key = f"{model_type}_{shape[0]}x{shape[1]}"
            all_results[result_key] = {
                'model_type': model_type,
                'matrix_shape': shape,
                'parameters': model.total_params,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1],
                'inference_time_ms': inference_time,
                'throughput_samples_per_sec': throughput,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            print(f"\nResults for {model_type} with shape {shape}:")
            print(f"  Best validation MSE: {best_val_loss:.6f}")
            print(f"  Inference time: {inference_time:.2f} ms")
            print(f"  Throughput: {throughput:.0f} samples/sec")
            print(f"  GCN baseline MSE: 0.006697")
            print(f"  Performance ratio: {best_val_loss/0.006697:.2f}x")
    
    # Save all results
    with open(os.path.join(save_dir, 'all_results.json'), 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ['train_losses', 'val_losses']} 
                  for k, v in all_results.items()}, f, indent=2)
    
    # Create comparison plot
    create_comparison_plot(all_results, save_dir)
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    
    # Sort by validation loss
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_val_loss'])
    
    print("\nModel Rankings (by validation MSE):")
    print(f"{'Rank':<5} {'Model':<25} {'Val MSE':<12} {'Params':<10} {'Speed (ms)':<12}")
    print("-" * 70)
    
    for i, (key, res) in enumerate(sorted_results[:5], 1):
        print(f"{i:<5} {key:<25} {res['best_val_loss']:<12.6f} "
              f"{res['parameters']:<10,} {res['inference_time_ms']:<12.2f}")
    
    print(f"\nGCN Baseline: MSE=0.006697, Params=50,689")
    
    best_model = sorted_results[0][1]
    print(f"\nBest Dense Model vs GCN:")
    print(f"  MSE ratio: {best_model['best_val_loss']/0.006697:.2f}x")
    print(f"  Parameter reduction: {(1-best_model['parameters']/50689)*100:.1f}%")
    
    print(f"\nAll results saved to: {save_dir}")


def create_comparison_plot(all_results, save_dir):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    model_names = list(all_results.keys())
    val_losses = [r['best_val_loss'] for r in all_results.values()]
    params = [r['parameters'] for r in all_results.values()]
    speeds = [r['inference_time_ms'] for r in all_results.values()]
    
    # Plot 1: Validation loss comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(model_names)), val_losses, color='steelblue')
    ax1.axhline(y=0.006697, color='red', linestyle='--', label='GCN Baseline')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('Validation MSE')
    ax1.set_title('Model Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Parameters comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(model_names)), params, color='orange')
    ax2.axhline(y=50689, color='red', linestyle='--', label='GCN Parameters')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Model Size Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training curves for best model
    ax3 = axes[1, 0]
    best_key = min(all_results.keys(), key=lambda k: all_results[k]['best_val_loss'])
    best_result = all_results[best_key]
    epochs = range(1, len(best_result['train_losses']) + 1)
    ax3.plot(epochs, best_result['train_losses'], 'b-', label='Train', alpha=0.7)
    ax3.plot(epochs, best_result['val_losses'], 'r-', label='Validation', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE Loss')
    ax3.set_title(f'Training Curves - Best Model ({best_key})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Speed vs Accuracy
    ax4 = axes[1, 1]
    ax4.scatter(speeds, val_losses, s=100, alpha=0.6)
    for i, name in enumerate(model_names):
        ax4.annotate(name, (speeds[i], val_losses[i]), fontsize=8, ha='center')
    ax4.set_xlabel('Inference Time (ms)')
    ax4.set_ylabel('Validation MSE')
    ax4.set_title('Speed vs Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Dense Matrix SpMV Models - Comprehensive Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'comparison_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {os.path.join(save_dir, 'comparison_plots.png')}")


if __name__ == "__main__":
    main()
