#!/usr/bin/env python3
"""
Create visualization plots for Dense Matrix SpMV models.
Investigate the suspiciously good performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.tri as mtri
from scipy.sparse import load_npz
from datetime import datetime
import os
import json

# Import the dense model architectures
from train_dense_spmv import DenseSpMV_TwoMatrix, DenseSpMV_Adaptive


def load_best_dense_models():
    """Load the best performing dense models."""
    
    # Find the latest results directory
    import glob
    result_dirs = glob.glob('dense_spmv_results_*')
    if not result_dirs:
        raise ValueError("No dense model results found!")
    
    latest_dir = sorted(result_dirs)[-1]
    print(f"Loading models from: {latest_dir}")
    
    # Load the results summary
    with open(os.path.join(latest_dir, 'all_results.json'), 'r') as f:
        results = json.load(f)
    
    # Load best models
    models = {}
    
    # Try to load all available two_matrix models
    two_matrix_configs = [(2, 324), (3, 216), (4, 162), (6, 108)]
    
    for m, n in two_matrix_configs:
        model_file = f'two_matrix_{m}x{n}_best.pth'
        model_path = os.path.join(latest_dir, model_file)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model = DenseSpMV_TwoMatrix(vector_dim=648, matrix_shape=(m, n))
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[f'two_matrix_{m}x{n}'] = (model, checkpoint['val_loss'])
                print(f"  Loaded: two_matrix_{m}x{n} (val_loss={checkpoint['val_loss']:.6f})")
            except Exception as e:
                print(f"  Warning: Could not load {model_file}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded!")
    
    return models, latest_dir


def load_data():
    """Load validation data."""
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250912_012326.npz')
    
    X_val = torch.FloatTensor(data['X_val'])
    Y_val = torch.FloatTensor(data['Y_val'])
    ks_val = data['ks_val']
    
    # For 2D plotting
    points = data['points']
    triangles = data['triangles']
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    
    return X_val, Y_val, ks_val, triang, data


def analyze_predictions(model, X_val, Y_val, model_name):
    """Analyze model predictions in detail."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    all_mse = []
    all_mae = []
    all_max_error = []
    all_relative_error = []
    
    with torch.no_grad():
        for i in range(len(X_val)):
            x_sample = X_val[i:i+1]  # Keep batch dimension
            y_true = Y_val[i]
            
            y_pred = model(x_sample).squeeze()
            
            # Calculate metrics
            mse = ((y_true - y_pred) ** 2).mean().item()
            mae = torch.abs(y_true - y_pred).mean().item()
            max_error = torch.abs(y_true - y_pred).max().item()
            
            # Relative error (avoiding division by zero)
            rel_error = torch.abs(y_true - y_pred) / (torch.abs(y_true) + 1e-10)
            mean_rel_error = rel_error.mean().item()
            
            all_mse.append(mse)
            all_mae.append(mae)
            all_max_error.append(max_error)
            all_relative_error.append(mean_rel_error)
    
    # Print statistics
    print(f"MSE: mean={np.mean(all_mse):.6f}, std={np.std(all_mse):.6f}")
    print(f"MAE: mean={np.mean(all_mae):.6f}, std={np.std(all_mae):.6f}")
    print(f"Max Error: mean={np.mean(all_max_error):.6f}, max={np.max(all_max_error):.6f}")
    print(f"Relative Error: mean={np.mean(all_relative_error):.6f}")
    
    # Check for suspiciously perfect predictions
    perfect_predictions = sum(1 for mse in all_mse if mse < 1e-10)
    print(f"Near-perfect predictions (MSE < 1e-10): {perfect_predictions}/{len(all_mse)}")
    
    return all_mse, all_mae, all_max_error


def create_detailed_comparison_plots(models, X_val, Y_val, ks_val, triang, save_dir):
    """Create detailed comparison plots for dense models."""
    
    # Select diverse samples
    np.random.seed(42)
    sample_indices = [0, 10, 25, 50, 75]  # Different from GNN visualization
    
    print(f"\nCreating visualizations for samples: {sample_indices}")
    
    for sample_idx in sample_indices:
        print(f"\nProcessing Sample {sample_idx}:")
        
        x_sample = X_val[sample_idx:sample_idx+1]
        y_true = Y_val[sample_idx]
        k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
        
        # Create figure with subplots for all models
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sample {sample_idx}: Dense Model Predictions (k1={k1}, k2={k2})', 
                    fontsize=16, fontweight='bold')
        
        # Ground truth
        ax = axes[0, 0]
        ax.plot(y_true.numpy(), 'k-', linewidth=2, label='Ground Truth')
        ax.set_title('Ground Truth K@f', fontsize=14, fontweight='bold')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Model predictions
        model_positions = [
            ('two_matrix_2x324', axes[0, 1]),
            ('two_matrix_3x216', axes[0, 2]),
            ('simple_3x216', axes[1, 0])
        ]
        
        for (model_name, ax) in model_positions:
            if model_name in models:
                model, val_loss = models[model_name]
                
                with torch.no_grad():
                    y_pred = model(x_sample).squeeze()
                
                # Calculate error
                mse = ((y_true - y_pred) ** 2).mean().item()
                mae = torch.abs(y_true - y_pred).mean().item()
                
                # Plot overlay
                ax.plot(y_true.numpy(), 'r-', linewidth=1.5, alpha=0.7, label='Ground Truth')
                ax.plot(y_pred.numpy(), 'g--', linewidth=1.5, alpha=0.7, label='Prediction')
                ax.set_title(f'{model_name}\nMSE={mse:.2e}, MAE={mae:.2e}', 
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Node Index')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Error analysis for best model
        ax = axes[1, 1]
        model, _ = models['two_matrix_2x324']
        with torch.no_grad():
            y_pred = model(x_sample).squeeze()
        
        error = (y_true - y_pred).numpy()
        ax.plot(error, 'b-', linewidth=1, alpha=0.8)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'Error: Best Model (two_matrix_2x324)\nMax={np.abs(error).max():.2e}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Error')
        ax.grid(True, alpha=0.3)
        
        # Histogram of errors
        ax = axes[1, 2]
        ax.hist(error, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        ax.set_title(f'Error Distribution\nStd={np.std(error):.2e}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(save_dir, f'dense_sample_{sample_idx:03d}_k{k1}k{k2}_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


def create_2d_field_comparison(models, X_val, Y_val, ks_val, triang, save_dir):
    """Create 2D field visualizations for dense models."""
    
    # Pick one sample for detailed 2D analysis
    sample_idx = 10
    x_sample = X_val[sample_idx:sample_idx+1]
    y_true = Y_val[sample_idx]
    input_vec = X_val[sample_idx].numpy()
    k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
    
    print(f"\nCreating 2D field plots for sample {sample_idx} (k1={k1}, k2={k2})")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Input field
    ax = axes[0, 0]
    cf = ax.tricontourf(triang, input_vec, levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_title(f'Input: sin(2π·{k1}·x)×sin(2π·{k2}·y)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Ground truth
    ax = axes[0, 1]
    cf = ax.tricontourf(triang, y_true.numpy(), levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_title('Ground Truth K@f', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Model predictions
    model_positions = [
        ('two_matrix_2x324', axes[0, 2]),
        ('two_matrix_3x216', axes[1, 0]),
        ('simple_3x216', axes[1, 1])
    ]
    
    for (model_name, ax) in model_positions:
        if model_name in models:
            model, _ = models[model_name]
            
            with torch.no_grad():
                y_pred = model(x_sample).squeeze().numpy()
            
            cf = ax.tricontourf(triang, y_pred, levels=20, cmap='viridis')
            plt.colorbar(cf, ax=ax, fraction=0.046)
            
            mse = np.mean((y_true.numpy() - y_pred) ** 2)
            ax.set_title(f'{model_name}\nMSE={mse:.2e}', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
    
    # Error field for best model
    ax = axes[1, 2]
    model, _ = models['two_matrix_2x324']
    with torch.no_grad():
        y_pred = model(x_sample).squeeze().numpy()
    
    error = np.abs(y_true.numpy() - y_pred)
    cf = ax.tricontourf(triang, error, levels=20, cmap='Reds')
    plt.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_title(f'Absolute Error (Best Model)\nMax={error.max():.2e}', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.suptitle(f'2D Field Comparison - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'dense_2d_fields_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved 2D field comparison: {save_path}")
    plt.close()


def investigate_perfect_predictions(models, X_val, Y_val, save_dir):
    """Investigate why predictions are suspiciously perfect."""
    
    print("\n" + "="*60)
    print("INVESTIGATING SUSPICIOUSLY PERFECT PREDICTIONS")
    print("="*60)
    
    model, val_loss = models['two_matrix_2x324']
    
    # Check weight matrices
    W1 = model.W1.detach().numpy()
    W2 = model.W2.detach().numpy()
    bias = model.bias.detach().numpy()
    
    print(f"\nModel: two_matrix_2x324")
    print(f"W1 shape: {W1.shape}, range: [{W1.min():.3f}, {W1.max():.3f}]")
    print(f"W2 shape: {W2.shape}, range: [{W2.min():.3f}, {W2.max():.3f}]")
    print(f"Bias range: [{bias.min():.3f}, {bias.max():.3f}]")
    
    # Check condition numbers
    cond_W1 = np.linalg.cond(W1)
    cond_W2 = np.linalg.cond(W2)
    print(f"Condition number W1: {cond_W1:.2e}")
    print(f"Condition number W2: {cond_W2:.2e}")
    
    # Visualize weight matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # W1 matrix
    im1 = axes[0].imshow(W1, cmap='RdBu_r', aspect='auto')
    axes[0].set_title(f'W1 Matrix (2×2)\nCond={cond_W1:.2e}')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])
    
    # W2 matrix (show subset for visibility)
    W2_subset = W2[:50, :50] if W2.shape[0] > 50 else W2
    im2 = axes[1].imshow(W2_subset, cmap='RdBu_r', aspect='auto')
    axes[1].set_title(f'W2 Matrix (first 50×50 of {W2.shape[0]}×{W2.shape[1]})')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1])
    
    # Bias vector
    axes[2].plot(bias, 'b-', linewidth=1)
    axes[2].set_title('Bias Vector')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Weight Matrix Analysis - Two Matrix Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'weight_matrix_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nWeight analysis saved: {save_path}")
    plt.close()
    
    # Test on random noise
    print("\n" + "-"*40)
    print("Testing on random noise input:")
    random_input = torch.randn(5, 648)
    with torch.no_grad():
        random_output = model(random_input)
    
    print(f"Random input range: [{random_input.min():.3f}, {random_input.max():.3f}]")
    print(f"Random output range: [{random_output.min():.3f}, {random_output.max():.3f}]")
    print(f"Output std: {random_output.std():.6f}")
    
    # Check if model memorized or learned patterns
    print("\n" + "-"*40)
    print("Checking for memorization vs learning:")
    
    # Compare predictions on first 10 samples
    memorization_check = []
    with torch.no_grad():
        for i in range(10):
            x = X_val[i:i+1]
            y_true = Y_val[i]
            y_pred = model(x).squeeze()
            
            correlation = np.corrcoef(y_true.numpy(), y_pred.numpy())[0, 1]
            memorization_check.append(correlation)
            
            if i < 3:
                print(f"Sample {i}: Correlation = {correlation:.6f}")
    
    print(f"Mean correlation: {np.mean(memorization_check):.6f}")
    
    if np.mean(memorization_check) > 0.9999:
        print("⚠️ WARNING: Model appears to have MEMORIZED the training data!")
        print("This explains the near-zero MSE values.")
    
    return W1, W2, bias


def create_ood_tests_with_higher_k(models, save_dir):
    """Test models on higher k-value sine combinations with proper normalization."""
    
    print("\n" + "="*60)
    print("OUT-OF-DISTRIBUTION TESTING: HIGHER K-VALUE SINE COMBINATIONS")
    print("="*60)
    
    # Load the sparse matrix K and mesh data
    K_sparse = load_npz('ml_data/sparse_matrix_K_20250912_012326.npz')
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250912_012326.npz')
    points = data['points']
    
    print(f"Sparse matrix K shape: {K_sparse.shape}, non-zeros: {K_sparse.nnz}")
    
    # Get only two_matrix models
    two_matrix_models = {k: v for k, v in models.items() if 'two_matrix' in k}
    sorted_models = sorted(two_matrix_models.items(), key=lambda x: x[1][1])
    
    # Define OOD k-value ranges (training was k1,k2 ∈ {1,2,3,4,5})
    ood_k_ranges = {
        'k6-k8': (6, 8),
        'k9-k12': (9, 12),
        'k13-k16': (13, 16),
        'k17-k20': (17, 20)
    }
    
    n_nodes = len(points)
    n_samples = 10
    np.random.seed(123)
    
    results = {}
    
    for case_name, (k_min, k_max) in ood_k_ranges.items():
        print(f"\nTesting {case_name}...")
        
        # Generate OOD samples
        inputs = []
        k_pairs = []
        true_outputs = []
        
        for _ in range(n_samples):
            k1 = np.random.randint(k_min, k_max + 1)
            k2 = np.random.randint(k_min, k_max + 1)
            
            # Generate sinusoidal field
            x_field = np.zeros(n_nodes)
            for i, point in enumerate(points):
                x, y = point
                x_field[i] = np.sin(2 * np.pi * k1 * x) * np.sin(2 * np.pi * k2 * y)
            
            # Normalize input to [-1, 1]
            if np.max(np.abs(x_field)) > 0:
                x_field = x_field / np.max(np.abs(x_field))
            
            # Compute true SpMV with frequency-aware normalization
            y_true_raw = K_sparse.dot(x_field)
            frequency_factor = k1*k1 + k2*k2
            reference_frequency = 1*1 + 1*1  # k1=1, k2=1 reference
            y_true_normalized = y_true_raw * (reference_frequency / frequency_factor)
            
            inputs.append(x_field)
            k_pairs.append((k1, k2))
            true_outputs.append(y_true_normalized)
        
        true_outputs = np.array(true_outputs)
        print(f"  Frequency factors: {[k1*k1 + k2*k2 for k1, k2 in k_pairs[:3]]}...")
        print(f"  Normalized output range: [{true_outputs.min():.3f}, {true_outputs.max():.3f}]")
        
        # Test each model
        case_results = {}
        for model_name, (model, _) in sorted_models:
            errors = []
            
            with torch.no_grad():
                for i, x in enumerate(inputs):
                    x_tensor = torch.FloatTensor(x)
                    y_pred = model(x_tensor.unsqueeze(0)).squeeze().numpy()
                    error = y_pred - true_outputs[i]
                    errors.append(error)
            
            errors = np.array(errors)
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))
            
            case_results[model_name] = {'mse': mse, 'mae': mae}
            print(f"  {model_name}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        results[case_name] = case_results
    
    # Create OOD visualization
    create_ood_visualization(ood_k_ranges, results, save_dir)
    
    return results


def create_ood_visualization(ood_k_ranges, results, save_dir):
    """Create comprehensive OOD visualization plots."""
    
    # Extract data for plotting
    case_names = list(ood_k_ranges.keys())
    model_names = list(results[case_names[0]].keys())
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: MSE comparison across k-ranges
    ax1 = axes[0, 0]
    x_pos = np.arange(len(case_names))
    width = 0.8 / len(model_names)
    
    for i, model_name in enumerate(model_names):
        mse_values = [results[case][model_name]['mse'] for case in case_names]
        bars = ax1.bar(x_pos + i*width, mse_values, width, 
                      label=model_name.replace('two_matrix_', ''), alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, mse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('K-value Range')
    ax1.set_ylabel('MSE')
    ax1.set_title('OOD Performance: MSE by K-Range')
    ax1.set_xticks(x_pos + width * (len(model_names)-1) / 2)
    ax1.set_xticklabels(case_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: MAE comparison across k-ranges
    ax2 = axes[0, 1]
    for i, model_name in enumerate(model_names):
        mae_values = [results[case][model_name]['mae'] for case in case_names]
        bars = ax2.bar(x_pos + i*width, mae_values, width, 
                      label=model_name.replace('two_matrix_', ''), alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, mae_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('K-value Range')
    ax2.set_ylabel('MAE')
    ax2.set_title('OOD Performance: MAE by K-Range')
    ax2.set_xticks(x_pos + width * (len(model_names)-1) / 2)
    ax2.set_xticklabels(case_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Model comparison (average across all k-ranges)
    ax3 = axes[1, 0]
    avg_mse = []
    avg_mae = []
    
    for model_name in model_names:
        mse_vals = [results[case][model_name]['mse'] for case in case_names]
        mae_vals = [results[case][model_name]['mae'] for case in case_names]
        avg_mse.append(np.mean(mse_vals))
        avg_mae.append(np.mean(mae_vals))
    
    x_models = np.arange(len(model_names))
    width_models = 0.35
    
    bars1 = ax3.bar(x_models - width_models/2, avg_mse, width_models, 
                   label='Average MSE', alpha=0.8, color='steelblue')
    bars2 = ax3.bar(x_models + width_models/2, avg_mae, width_models, 
                   label='Average MAE', alpha=0.8, color='orange')
    
    # Add value labels
    for bar, val in zip(bars1, avg_mse):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, avg_mae):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Error')
    ax3.set_title('Average OOD Performance Across All K-Ranges')
    ax3.set_xticks(x_models)
    ax3.set_xticklabels([name.replace('two_matrix_', '') for name in model_names])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Heatmap of MSE values
    ax4 = axes[1, 1]
    mse_matrix = np.zeros((len(case_names), len(model_names)))
    
    for i, case in enumerate(case_names):
        for j, model in enumerate(model_names):
            mse_matrix[i, j] = results[case][model]['mse']
    
    im = ax4.imshow(mse_matrix, cmap='hot', aspect='auto')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels([name.replace('two_matrix_', '') for name in model_names])
    ax4.set_yticks(range(len(case_names)))
    ax4.set_yticklabels(case_names)
    ax4.set_title('OOD MSE Heatmap')
    
    # Add text annotations
    for i in range(len(case_names)):
        for j in range(len(model_names)):
            text = ax4.text(j, i, f'{mse_matrix[i, j]:.3f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.suptitle('Out-of-Distribution Performance Analysis\nHigher K-Value Sine Combinations with Frequency-Aware Normalization', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'ood_performance_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ OOD performance analysis saved to: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("DENSE MATRIX MODEL VISUALIZATION & INVESTIGATION")
    print("="*80)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"dense_model_plots_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load models and data
    models, results_dir = load_best_dense_models()
    X_val, Y_val, ks_val, triang, data = load_data()
    
    # Analyze each model
    all_metrics = {}
    for model_name, (model, val_loss) in models.items():
        print(f"\nModel: {model_name}, Reported Val Loss: {val_loss:.6f}")
        mse_list, mae_list, max_err_list = analyze_predictions(model, X_val, Y_val, model_name)
        all_metrics[model_name] = {
            'mse': mse_list,
            'mae': mae_list,
            'max_error': max_err_list
        }
    
    # Create visualizations
    create_detailed_comparison_plots(models, X_val, Y_val, ks_val, triang, save_dir)
    create_2d_field_comparison(models, X_val, Y_val, ks_val, triang, save_dir)
    
    # Test on higher k-values with proper normalization
    ood_results = create_ood_tests_with_higher_k(models, save_dir)
    
    # Investigate perfect predictions
    W1, W2, bias = investigate_perfect_predictions(models, X_val, Y_val, save_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY WITH FREQUENCY-AWARE NORMALIZATION")
    print("="*60)
    
    print("\n--- In-Distribution Performance ---")
    for model_name in all_metrics:
        metrics = all_metrics[model_name]
        print(f"\n{model_name}:")
        print(f"  Mean MSE: {np.mean(metrics['mse']):.6e}")
        print(f"  Mean MAE: {np.mean(metrics['mae']):.6e}")
        print(f"  Mean Max Error: {np.mean(metrics['max_error']):.6e}")
    
    print("\n--- Out-of-Distribution Performance (Higher K-Values) ---")
    for case_name in ood_results:
        print(f"\n{case_name}:")
        for model_name in sorted(ood_results[case_name].keys()):
            mse = ood_results[case_name][model_name]['mse']
            mae = ood_results[case_name][model_name]['mae']
            print(f"  {model_name}: MSE={mse:.6f}, MAE={mae:.6f}")
    
    print(f"\n✅ All visualizations saved to: {save_dir}")
    print("\n⚠️ Check the plots carefully for signs of overfitting or memorization!")
    
    # Save metrics
    import json
    with open(os.path.join(save_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump({k: {kk: float(np.mean(vv)) for kk, vv in v.items()} 
                  for k, v in all_metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
