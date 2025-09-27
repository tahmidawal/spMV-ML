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
import matplotlib.gridspec as gridspec
from datetime import datetime
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class DenseSpMV_TwoMatrix(nn.Module):
    """
    Two-matrix approach: Y = W1 @ X @ W2^T
    More expressive with two learnable matrices
    """
    def __init__(self, vector_dim=1424, matrix_shape=(16, 89)):
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


class DenseSpMV_SingleMatrix(nn.Module):
    """
    Single matrix approach: Y = W @ X_reshaped
    More parameter-efficient for large vectors
    """
    def __init__(self, vector_dim=1424, matrix_shape=(16, 89)):
        super().__init__()
        self.vector_dim = vector_dim
        self.m, self.n = matrix_shape
        assert self.m * self.n == vector_dim
        
        # Single transformation matrix
        self.W = nn.Parameter(torch.randn(self.m, self.m) * 0.02)
        self.bias = nn.Parameter(torch.zeros(vector_dim))
        
        self.total_params = self.m * self.m + vector_dim
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Reshape to matrix
        X_in = x.view(batch_size, self.m, self.n)
        
        # Single matrix multiplication: W @ X
        X_out = torch.bmm(self.W.unsqueeze(0).expand(batch_size, -1, -1), X_in)
        
        # Reshape back
        y = X_out.view(batch_size, self.vector_dim)
        y = y + self.bias
        
        return y.squeeze() if batch_size == 1 else y


class DenseSpMV_Adaptive(nn.Module):
    """
    Adaptive approach with lightweight preprocessing and postprocessing
    """
    def __init__(self, vector_dim=1424, matrix_shape=(16, 89), hidden_dim=128):
        super().__init__()
        self.vector_dim = vector_dim
        self.m, self.n = matrix_shape
        assert self.m * self.n == vector_dim
        
        # Lightweight preprocessing layer
        self.input_proj = nn.Linear(vector_dim, vector_dim)
        
        # Main transformation matrices
        self.W1 = nn.Parameter(torch.randn(self.m, self.m) * 0.02)
        self.W2 = nn.Parameter(torch.randn(self.n, self.n) * 0.02)
        
        # Lightweight postprocessing
        self.output_proj = nn.Linear(vector_dim, vector_dim)
        
        self.total_params = (vector_dim * vector_dim + vector_dim +  # input_proj
                           self.m * self.m + self.n * self.n +        # W1, W2
                           vector_dim * vector_dim + vector_dim)      # output_proj
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Input preprocessing
        x = self.input_proj(x)
        
        # Reshape to matrix
        X_in = x.view(batch_size, self.m, self.n)
        
        # Two matrix multiplications: W1 @ X @ W2^T
        X_temp = torch.bmm(self.W1.unsqueeze(0).expand(batch_size, -1, -1), X_in)
        X_out = torch.bmm(X_temp, self.W2.t().unsqueeze(0).expand(batch_size, -1, -1))
        
        # Reshape back
        y = X_out.view(batch_size, self.vector_dim)
        
        # Output postprocessing
        y = self.output_proj(y)
        
        return y.squeeze() if batch_size == 1 else y


def find_optimal_shapes(dim=1424):
    """Find all valid matrix shapes for reshaping."""
    shapes = []
    for m in range(2, int(np.sqrt(dim)) + 1):
        if dim % m == 0:
            n = dim // m
            shapes.append((m, n))
    return shapes


def load_data():
    """Load the FEM Poisson SpMV dataset."""
    print("Loading FEM Poisson dataset...")
    
    # Load training data
    train_data = np.load('ml_data/order1_mixed_1500samples.npz', allow_pickle=True)
    inputs = train_data['inputs']  # Shape: (1500, 1424)
    outputs = train_data['outputs']  # Shape: (1500, 1424)
    
    # Convert to torch tensors and ensure float32
    inputs = torch.FloatTensor(inputs.astype(np.float32))
    outputs = torch.FloatTensor(outputs.astype(np.float32))
    
    # Split into train/validation (80/20 split)
    n_samples = inputs.shape[0]
    n_train = int(0.8 * n_samples)
    
    # Create random indices for train/val split
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train = inputs[train_indices]
    Y_train = outputs[train_indices]
    X_val = inputs[val_indices]
    Y_val = outputs[val_indices]
    
    print(f"Dataset loaded:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Vector dimension: {X_train.shape[1]}")
    print(f"  Input range: [{X_train.min():.6f}, {X_train.max():.6f}]")
    print(f"  Output range: [{Y_train.min():.6f}, {Y_train.max():.6f}]")
    
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


def create_validation_plots(model, X_val, Y_val, model_name, save_dir, device='cpu'):
    """Create validation plots comparing predictions vs actual outputs."""
    model.eval()
    
    print(f"   Creating validation plots for {model_name}...")
    
    with torch.no_grad():
        # Get predictions for validation set
        batch_size = 50
        n_samples = X_val.shape[0]
        predictions = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = X_val[start_idx:end_idx].to(device)
            pred_batch = model(x_batch)
            predictions.append(pred_batch.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        Y_val_np = Y_val.cpu().numpy()
        predictions_np = predictions.numpy()
    
    # Create comprehensive validation plots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)
    
    # Select random samples to plot
    n_plot_samples = 6
    sample_indices = np.random.choice(n_samples, n_plot_samples, replace=False)
    
    # Plot 1-6: Individual sample comparisons (2 rows, 3 columns)
    for i, sample_idx in enumerate(sample_indices):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        actual = Y_val_np[sample_idx]
        predicted = predictions_np[sample_idx]
        
        ax.plot(actual, 'b-', linewidth=1, alpha=0.8, label='Actual')
        ax.plot(predicted, 'r--', linewidth=1, alpha=0.8, label='Predicted')
        ax.set_title(f'Sample {sample_idx}: Prediction vs Actual', fontsize=10, fontweight='bold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add error statistics
        mse = np.mean((actual - predicted)**2)
        mae = np.mean(np.abs(actual - predicted))
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
    
    # Plot 7: Overall prediction vs actual scatter
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Sample points for scatter plot (too many points would be cluttered)
    sample_points = np.random.choice(Y_val_np.size, min(5000, Y_val_np.size), replace=False)
    actual_flat = Y_val_np.flatten()[sample_points]
    pred_flat = predictions_np.flatten()[sample_points]
    
    ax7.scatter(actual_flat, pred_flat, alpha=0.3, s=1)
    
    # Add perfect prediction line
    min_val = min(np.min(actual_flat), np.min(pred_flat))
    max_val = max(np.max(actual_flat), np.max(pred_flat))
    ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax7.set_xlabel('Actual Values')
    ax7.set_ylabel('Predicted Values')
    ax7.set_title('Prediction vs Actual (All Values)', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Calculate R²
    ss_res = np.sum((actual_flat - pred_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    ax7.text(0.02, 0.98, f'R² = {r2:.4f}', transform=ax7.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    
    # Plot 8: Error distribution
    ax8 = fig.add_subplot(gs[2, 1])
    errors = predictions_np - Y_val_np
    ax8.hist(errors.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax8.set_xlabel('Prediction Error')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Error Distribution', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Add error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax8.text(0.02, 0.98, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
             transform=ax8.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
    
    # Plot 9: Per-sample MSE
    ax9 = fig.add_subplot(gs[2, 2])
    sample_mses = np.mean((predictions_np - Y_val_np)**2, axis=1)
    ax9.hist(sample_mses, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax9.axvline(np.mean(sample_mses), color='red', linestyle='--', linewidth=2, 
                label=f'Mean MSE: {np.mean(sample_mses):.4f}')
    ax9.set_xlabel('Per-Sample MSE')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Per-Sample MSE Distribution', fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Norm comparison
    ax10 = fig.add_subplot(gs[2, 3])
    actual_norms = np.linalg.norm(Y_val_np, axis=1)
    pred_norms = np.linalg.norm(predictions_np, axis=1)
    
    ax10.scatter(actual_norms, pred_norms, alpha=0.6, s=20)
    min_norm = min(np.min(actual_norms), np.min(pred_norms))
    max_norm = max(np.max(actual_norms), np.max(pred_norms))
    ax10.plot([min_norm, max_norm], [min_norm, max_norm], 'r--', linewidth=2, label='Perfect Match')
    
    ax10.set_xlabel('Actual ||y||₂')
    ax10.set_ylabel('Predicted ||y||₂')
    ax10.set_title('Output Norm Comparison', fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Calculate correlation for norms
    norm_corr = np.corrcoef(actual_norms, pred_norms)[0, 1]
    ax10.text(0.02, 0.98, f'Corr: {norm_corr:.4f}', transform=ax10.transAxes, va='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)
    
    # Overall title
    overall_mse = np.mean((predictions_np - Y_val_np)**2)
    overall_mae = np.mean(np.abs(predictions_np - Y_val_np))
    plt.suptitle(f'Validation Results: {model_name}\n'
                 f'Overall MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}, R²: {r2:.4f}', 
                 fontsize=14, fontweight='bold')
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{model_name}_validation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Validation plots saved to: {plot_path}")
    
    # Return validation metrics (convert to Python floats for JSON serialization)
    return {
        'overall_mse': float(overall_mse),
        'overall_mae': float(overall_mae),
        'r2_score': float(r2),
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'norm_correlation': float(norm_corr)
    }


def create_ood_signal(n, signal_type='gaussian', seed=None):
    """Create out-of-distribution signals for validation"""
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(0, 1, n)
    
    if signal_type == 'gaussian':
        # Gaussian random noise with zero boundaries
        signal = np.random.normal(0, 0.5, n)
        # Apply boundary conditions
        signal[0] = 0
        signal[-1] = 0
        # Smooth transition to boundaries
        boundary_width = n // 20
        for i in range(boundary_width):
            weight = i / boundary_width
            signal[i] *= weight
            signal[-(i+1)] *= weight
        info = {'type': 'gaussian', 'std': 0.5}
        
    elif signal_type == 'uniform':
        # Uniform random noise with zero boundaries
        signal = np.random.uniform(-1, 1, n)
        signal[0] = 0
        signal[-1] = 0
        # Smooth transition to boundaries
        boundary_width = n // 20
        for i in range(boundary_width):
            weight = i / boundary_width
            signal[i] *= weight
            signal[-(i+1)] *= weight
        info = {'type': 'uniform', 'range': [-1, 1]}
        
    elif signal_type == 'polynomial':
        # Random polynomial with zero boundaries
        degree = np.random.randint(2, 6)
        coeffs = np.random.uniform(-2, 2, degree)
        
        signal = np.zeros(n)
        for i, coeff in enumerate(coeffs):
            signal += coeff * (x ** (i + 1))
        
        # Normalize and apply boundary conditions
        signal = signal / np.max(np.abs(signal)) * np.random.uniform(0.5, 1.0)
        signal[0] = 0
        signal[-1] = 0
        info = {'type': 'polynomial', 'degree': degree, 'coeffs': coeffs.tolist()}
        
    elif signal_type == 'exponential':
        # Exponential decay/growth with zero boundaries
        alpha = np.random.uniform(-3, 3)
        beta = np.random.uniform(0.5, 2)
        signal = np.exp(alpha * x) * np.sin(beta * np.pi * x)
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * np.random.uniform(0.5, 1.0)
        signal[0] = 0
        signal[-1] = 0
        info = {'type': 'exponential', 'alpha': alpha, 'beta': beta}
        
    elif signal_type == 'step':
        # Step function with random positions
        n_steps = np.random.randint(2, 6)
        step_positions = np.sort(np.random.uniform(0.1, 0.9, n_steps))
        step_values = np.random.uniform(-1, 1, n_steps + 1)
        
        signal = np.zeros(n)
        for i in range(n):
            if x[i] < step_positions[0]:
                signal[i] = step_values[0]
            else:
                for j in range(len(step_positions) - 1):
                    if step_positions[j] <= x[i] < step_positions[j + 1]:
                        signal[i] = step_values[j + 1]
                        break
                else:
                    signal[i] = step_values[-1]
        
        # Apply boundary conditions
        signal[0] = 0
        signal[-1] = 0
        info = {'type': 'step', 'n_steps': n_steps, 'positions': step_positions.tolist()}
        
    elif signal_type == 'sawtooth':
        # Sawtooth wave with zero boundaries
        freq = np.random.uniform(2, 20)
        signal = 2 * (x * freq - np.floor(x * freq + 0.5))
        
        # Normalize and apply boundary conditions
        signal = signal / np.max(np.abs(signal)) * np.random.uniform(0.5, 1.0)
        signal[0] = 0
        signal[-1] = 0
        info = {'type': 'sawtooth', 'freq': freq}
        
    elif signal_type == 'mixed':
        # Mixed random components
        components = []
        n_comp = np.random.randint(2, 4)
        
        for _ in range(n_comp):
            comp_type = np.random.choice(['sin', 'cos', 'exp', 'poly'])
            if comp_type == 'sin':
                freq = np.random.uniform(1, 30)
                amp = np.random.uniform(0.2, 0.8)
                components.append(amp * np.sin(2 * np.pi * freq * x))
            elif comp_type == 'cos':
                freq = np.random.uniform(1, 30)
                amp = np.random.uniform(0.2, 0.8)
                components.append(amp * np.cos(2 * np.pi * freq * x))
            elif comp_type == 'exp':
                alpha = np.random.uniform(-2, 2)
                amp = np.random.uniform(0.2, 0.8)
                components.append(amp * np.exp(alpha * x))
            else:  # poly
                degree = np.random.randint(1, 4)
                coeffs = np.random.uniform(-1, 1, degree)
                poly = np.zeros(n)
                for i, coeff in enumerate(coeffs):
                    poly += coeff * (x ** (i + 1))
                amp = np.random.uniform(0.2, 0.8)
                components.append(amp * poly)
        
        signal = np.sum(components, axis=0)
        
        # Normalize and apply boundary conditions
        signal = signal / np.max(np.abs(signal)) * np.random.uniform(0.5, 1.0)
        signal[0] = 0
        signal[-1] = 0
        info = {'type': 'mixed', 'n_components': n_comp}
    
    else:  # 'random'
        # Randomly choose from OOD types
        ood_types = ['gaussian', 'uniform', 'polynomial', 'exponential', 'step', 'sawtooth', 'mixed']
        chosen_type = np.random.choice(ood_types)
        return create_ood_signal(n, chosen_type, seed)
    
    return signal, info


def load_sparse_matrix_for_ood():
    """Load the sparse matrix for OOD validation"""
    import sys
    import os
    sys.path.append('/Users/tahmidawal/sparse2dense/0912-SpMV-CNN2D-SparseMatrices-nonorm/fem-poisson')
    
    import poisson_utils.preprocess
    import global_assembl_poisson.poisson
    from scipy.sparse import csr_matrix
    import jax
    import jax.numpy as jnp
    
    # Configuration for ORDER 1 (same as training data)
    p_order = 1
    mesh_file_path = "/Users/tahmidawal/sparse2dense/fem-poisson/unstructured-tubev2-1050.msh"
    surface_tags = [31]
    
    # Load and assemble matrix
    all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n, e_to_n_sort_idx, e_to_n_sorted, \
        bdry_indices, refel, u_exact = poisson_utils.preprocess.preprocess_poisson(
            p_order, mesh_file_path, surface_tags
        )
    
    n_pts = all_pts_gll_x.shape[0]
    
    jax_sparse_mat = global_assembl_poisson.poisson.assemble_global(
        p_order, all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n,
        refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d
    )
    
    scipy_sparse = csr_matrix(
        (np.array(jax_sparse_mat.data), 
         np.array(jax_sparse_mat.indices), 
         np.array(jax_sparse_mat.indptr)),
        shape=(n_pts, n_pts)
    )
    
    return scipy_sparse


def generate_ood_data(vector_dim=1424, n_ood_samples=200):
    """Generate out-of-distribution test data with true sparse matrix outputs"""
    print(f"\nGenerating {n_ood_samples} OOD samples...")
    
    # Load sparse matrix for computing true outputs
    print("Loading sparse matrix for OOD validation...")
    sparse_matrix = load_sparse_matrix_for_ood()
    print(f"Loaded sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz:,} non-zeros")
    
    # OOD signal types to test
    ood_types = ['gaussian', 'uniform', 'polynomial', 'exponential', 'step', 'sawtooth', 'mixed']
    samples_per_type = n_ood_samples // len(ood_types)
    
    ood_inputs = []
    ood_outputs = []  # True sparse matrix outputs
    ood_infos = []
    
    for ood_type in ood_types:
        for i in range(samples_per_type):
            signal, info = create_ood_signal(vector_dim, ood_type, seed=40000 + len(ood_inputs))
            
            # Compute true sparse matrix output
            true_output = sparse_matrix @ signal
            
            ood_inputs.append(signal)
            ood_outputs.append(true_output)
            ood_infos.append(info)
    
    # Fill remaining slots with random types
    remaining = n_ood_samples - len(ood_inputs)
    for i in range(remaining):
        signal, info = create_ood_signal(vector_dim, 'random', seed=50000 + i)
        
        # Compute true sparse matrix output
        true_output = sparse_matrix @ signal
        
        ood_inputs.append(signal)
        ood_outputs.append(true_output)
        ood_infos.append(info)
    
    ood_inputs = np.array(ood_inputs)
    ood_outputs = np.array(ood_outputs)
    
    # Count signal types
    type_counts = {}
    for info in ood_infos:
        t = info['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"OOD signal type distribution:")
    for t, count in type_counts.items():
        print(f"  {t}: {count} ({100*count/len(ood_infos):.1f}%)")
    
    print(f"OOD data ranges:")
    print(f"  Input range: [{np.min(ood_inputs):.6f}, {np.max(ood_inputs):.6f}]")
    print(f"  Output range: [{np.min(ood_outputs):.6f}, {np.max(ood_outputs):.6f}]")
    
    return torch.FloatTensor(ood_inputs), torch.FloatTensor(ood_outputs), ood_infos


def create_ood_validation_plots(model, ood_inputs, ood_outputs, ood_infos, model_name, save_dir, device='cpu'):
    """Create OOD validation plots comparing different signal types"""
    model.eval()
    
    print(f"   Creating OOD validation plots for {model_name}...")
    
    with torch.no_grad():
        # Get predictions for OOD data
        batch_size = 50
        n_samples = ood_inputs.shape[0]
        predictions = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = ood_inputs[start_idx:end_idx].to(device)
            pred_batch = model(x_batch)
            predictions.append(pred_batch.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        ood_inputs_np = ood_inputs.cpu().numpy()
        ood_outputs_np = ood_outputs.cpu().numpy()  # True sparse matrix outputs
        predictions_np = predictions.numpy()
    
    # Create comprehensive OOD validation plots
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.3)
    
    # Get unique signal types and select samples
    unique_types = list(set(info['type'] for info in ood_infos))
    selected_samples = []
    
    # Select 2 samples from each type (up to 12 samples total)
    for signal_type in unique_types[:6]:  # Limit to 6 types
        type_indices = [i for i, info in enumerate(ood_infos) if info['type'] == signal_type]
        selected_samples.extend(np.random.choice(type_indices, min(2, len(type_indices)), replace=False))
    
    # Plot individual sample comparisons (3 rows, 4 columns = 12 plots)
    for i, sample_idx in enumerate(selected_samples[:12]):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        true_output = ood_outputs_np[sample_idx]  # True A·u
        predicted = predictions_np[sample_idx]    # Neural network prediction
        info = ood_infos[sample_idx]
        
        ax.plot(true_output, 'b-', linewidth=1, alpha=0.8, label='True A·u')
        ax.plot(predicted, 'r--', linewidth=1, alpha=0.8, label='Predicted')
        
        # Create title based on signal type
        if info['type'] == 'gaussian':
            title = f"Gaussian (σ={info['std']})"
        elif info['type'] == 'polynomial':
            title = f"Poly (deg={info['degree']})"
        elif info['type'] == 'exponential':
            title = f"Exp (α={info['alpha']:.2f})"
        elif info['type'] == 'step':
            title = f"Step ({info['n_steps']} steps)"
        elif info['type'] == 'sawtooth':
            title = f"Sawtooth (f={info['freq']:.1f})"
        elif info['type'] == 'mixed':
            title = f"Mixed ({info['n_components']} comp)"
        else:
            title = f"{info['type'].title()}"
        
        ax.set_title(f'OOD {title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add MSE between true output and prediction
        mse = np.mean((true_output - predicted)**2)
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
    
    # Plot 13: OOD performance by signal type
    ax13 = fig.add_subplot(gs[3, 0])
    
    # Calculate MSE by signal type (true output vs predicted)
    type_mses = {}
    for i, info in enumerate(ood_infos):
        signal_type = info['type']
        if signal_type not in type_mses:
            type_mses[signal_type] = []
        mse = np.mean((ood_outputs_np[i] - predictions_np[i])**2)
        type_mses[signal_type].append(mse)
    
    # Box plot of MSEs by type
    box_data = [type_mses[t] for t in sorted(type_mses.keys())]
    box_labels = sorted(type_mses.keys())
    ax13.boxplot(box_data, labels=box_labels)
    ax13.set_title('OOD MSE by Signal Type', fontweight='bold')
    ax13.set_xlabel('Signal Type')
    ax13.set_ylabel('MSE')
    ax13.tick_params(axis='x', rotation=45)
    ax13.grid(True, alpha=0.3)
    
    # Plot 14: Overall OOD error distribution
    ax14 = fig.add_subplot(gs[3, 1])
    all_errors = (ood_outputs_np - predictions_np).flatten()
    ax14.hist(all_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax14.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax14.set_xlabel('Prediction Error (True A·u - Predicted)')
    ax14.set_ylabel('Frequency')
    ax14.set_title('OOD Error Distribution', fontweight='bold')
    ax14.legend()
    ax14.grid(True, alpha=0.3)
    
    # Plot 15: OOD vs Training data comparison
    ax15 = fig.add_subplot(gs[3, 2])
    
    # Calculate overall statistics (true output vs predicted)
    overall_mse = np.mean((ood_outputs_np - predictions_np)**2)
    overall_mae = np.mean(np.abs(ood_outputs_np - predictions_np))
    
    # Create a simple bar comparison (placeholder - would need training data for real comparison)
    metrics = ['MSE', 'MAE']
    ood_values = [overall_mse, overall_mae]
    
    bars = ax15.bar(metrics, ood_values, color='orange', alpha=0.7)
    ax15.set_title('OOD Performance Metrics', fontweight='bold')
    ax15.set_ylabel('Value')
    ax15.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, ood_values):
        ax15.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 16: Output complexity vs error
    ax16 = fig.add_subplot(gs[3, 3])
    
    # Calculate output complexity (using standard deviation as proxy)
    complexities = []
    mses = []
    for i in range(len(ood_outputs_np)):
        complexity = np.std(ood_outputs_np[i])  # True output complexity
        mse = np.mean((ood_outputs_np[i] - predictions_np[i])**2)
        complexities.append(complexity)
        mses.append(mse)
    
    ax16.scatter(complexities, mses, alpha=0.6, s=20)
    ax16.set_xlabel('True Output Complexity (std)')
    ax16.set_ylabel('MSE')
    ax16.set_title('Output Complexity vs Error', fontweight='bold')
    ax16.grid(True, alpha=0.3)
    
    # Overall title with statistics
    plt.suptitle(f'OOD Validation Results: {model_name}\n'
                 f'Overall MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}, '
                 f'Samples: {len(ood_outputs_np)}', 
                 fontsize=14, fontweight='bold')
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{model_name}_ood_validation_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   OOD validation plots saved to: {plot_path}")
    
    # Return OOD validation metrics
    return {
        'ood_overall_mse': float(overall_mse),
        'ood_overall_mae': float(overall_mae),
        'ood_type_mses': {t: float(np.mean(mses)) for t, mses in type_mses.items()},
        'ood_n_samples': len(ood_outputs_np)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Dense Matrix SpMV models')
    parser.add_argument('--model', type=str, default='single_matrix', 
                       choices=['single_matrix', 'two_matrix', 'adaptive', 'all'])
    parser.add_argument('--epochs', type=int, default=500)
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
    
    # Generate OOD validation data
    X_ood, Y_ood, ood_infos = generate_ood_data(vector_dim=1424, n_ood_samples=200)
    
    # Find valid matrix shapes
    valid_shapes = find_optimal_shapes(1424)
    print(f"\nValid matrix shapes for 1424-dim vector: {valid_shapes}")
    
    # Determine shapes to try
    if args.matrix_shape == 'auto':
        # Include specific shapes we want to test for 1424 = 2^4 × 89
        shapes_to_try = [(2, 712), (4, 356), (8, 178), (16, 89)]
    else:
        m, n = map(int, args.matrix_shape.split(','))
        shapes_to_try = [(m, n)]
    
    print(f"Shapes to be tested: {shapes_to_try}")
    
    # Determine models to train
    if args.model == 'all':
        model_types = ['single_matrix', 'two_matrix', 'adaptive']
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
            if model_type == 'single_matrix':
                model = DenseSpMV_SingleMatrix(vector_dim=1424, matrix_shape=shape)
            elif model_type == 'two_matrix':
                model = DenseSpMV_TwoMatrix(vector_dim=1424, matrix_shape=shape)
            elif model_type == 'adaptive':
                model = DenseSpMV_Adaptive(vector_dim=1424, matrix_shape=shape)
            
            model = model.to(device)
            
            print(f"Model parameters: {model.total_params:,}")
            sparse_matrix_nnz = 31602  # From the FEM matrix
            print(f"Sparse matrix non-zeros: {sparse_matrix_nnz:,}")
            print(f"Parameter efficiency: {model.total_params/sparse_matrix_nnz:.2f}x sparse entries")
            
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
            
            # Create validation plots
            result_key = f"{model_type}_{shape[0]}x{shape[1]}"
            validation_metrics = create_validation_plots(model, X_val, Y_val, result_key, save_dir, device)
            
            # Create OOD validation plots
            ood_metrics = create_ood_validation_plots(model, X_ood, Y_ood, ood_infos, result_key, save_dir, device)
            
            # Store results
            all_results[result_key] = {
                'model_type': model_type,
                'matrix_shape': shape,
                'parameters': model.total_params,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_losses[-1],
                'inference_time_ms': inference_time,
                'throughput_samples_per_sec': throughput,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'validation_metrics': validation_metrics,
                'ood_metrics': ood_metrics
            }
            
            print(f"\nResults for {model_type} with shape {shape}:")
            print(f"  Best validation MSE: {best_val_loss:.6f}")
            print(f"  Validation R²: {validation_metrics['r2_score']:.4f}")
            print(f"  Validation MAE: {validation_metrics['overall_mae']:.6f}")
            print(f"  OOD MSE: {ood_metrics['ood_overall_mse']:.6f}")
            print(f"  OOD MAE: {ood_metrics['ood_overall_mae']:.6f}")
            print(f"  Inference time: {inference_time:.2f} ms")
            print(f"  Throughput: {throughput:.0f} samples/sec")
            print(f"  Parameter efficiency: {model.total_params/31602:.2f}x sparse matrix entries")
    
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
    print(f"{'Rank':<5} {'Model':<25} {'Val MSE':<12} {'OOD MSE':<12} {'R²':<8} {'Params':<10} {'Speed (ms)':<12}")
    print("-" * 92)
    
    for i, (key, res) in enumerate(sorted_results[:5], 1):
        r2_score = res.get('validation_metrics', {}).get('r2_score', 0.0)
        ood_mse = res.get('ood_metrics', {}).get('ood_overall_mse', 0.0)
        print(f"{i:<5} {key:<25} {res['best_val_loss']:<12.6f} {ood_mse:<12.6f} {r2_score:<8.4f} "
              f"{res['parameters']:<10,} {res['inference_time_ms']:<12.2f}")
    
    print(f"\nSparse Matrix: NNZ=31,602 entries")
    
    best_model = sorted_results[0][1]
    best_val_metrics = best_model.get('validation_metrics', {})
    best_ood_metrics = best_model.get('ood_metrics', {})
    print(f"\nBest Dense Model Analysis:")
    print(f"  Best Validation MSE: {best_model['best_val_loss']:.6f}")
    print(f"  R² Score: {best_val_metrics.get('r2_score', 0.0):.4f}")
    print(f"  Validation MAE: {best_val_metrics.get('overall_mae', 0.0):.6f}")
    print(f"  OOD MSE: {best_ood_metrics.get('ood_overall_mse', 0.0):.6f}")
    print(f"  OOD MAE: {best_ood_metrics.get('ood_overall_mae', 0.0):.6f}")
    print(f"  Norm Correlation: {best_val_metrics.get('norm_correlation', 0.0):.4f}")
    print(f"  Parameters: {best_model['parameters']:,}")
    print(f"  Parameter efficiency: {best_model['parameters']/31602:.2f}x sparse entries")
    
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
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('Validation MSE')
    ax1.set_title('Model Performance Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Parameters comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(model_names)), params, color='orange')
    ax2.axhline(y=31602, color='red', linestyle='--', label='Sparse Matrix NNZ')
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
