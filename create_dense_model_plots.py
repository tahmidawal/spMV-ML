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
    
    # Dynamically find all two_matrix models
    two_matrix_configs = [
        (2, 324),
        (3, 216),
        (4, 162),
        (6, 108)  # New configuration
    ]
    
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
    
    # Also try to load adaptive models if they exist
    for m, n in two_matrix_configs:
        model_file = f'adaptive_{m}x{n}_best.pth'
        model_path = os.path.join(latest_dir, model_file)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model = DenseSpMV_Adaptive(vector_dim=648, matrix_shape=(m, n))
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[f'adaptive_{m}x{n}'] = (model, checkpoint['val_loss'])
                print(f"  Loaded: adaptive_{m}x{n} (val_loss={checkpoint['val_loss']:.6f})")
            except Exception as e:
                print(f"  Warning: Could not load {model_file}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded!")
    
    return models, latest_dir


def load_data():
    """Load validation data."""
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    
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
    
    # Get only two_matrix models and sort them
    two_matrix_models = {k: v for k, v in models.items() if 'two_matrix' in k}
    sorted_models = sorted(two_matrix_models.items(), key=lambda x: x[1][1])  # Sort by val_loss
    
    for sample_idx in sample_indices:
        print(f"\nProcessing Sample {sample_idx}:")
        
        x_sample = X_val[sample_idx:sample_idx+1]
        y_true = Y_val[sample_idx]
        k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
        
        # Determine grid size based on number of models
        n_models = len(two_matrix_models)
        n_cols = 3
        n_rows = (n_models + 5) // n_cols  # +5 for ground truth, error plot, histogram
        
        # Create figure with subplots for all models
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Sample {sample_idx}: Two-Matrix Model Predictions (k1={k1}, k2={k2})', 
                    fontsize=16, fontweight='bold')
        
        # Ground truth
        ax = axes.flat[0]
        ax.plot(y_true.numpy(), 'k-', linewidth=2, label='Ground Truth')
        ax.set_title('Ground Truth K@f', fontsize=14, fontweight='bold')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Model predictions
        for idx, (model_name, (model, val_loss)) in enumerate(sorted_models):
            if idx + 1 < len(axes.flat):
                ax = axes.flat[idx + 1]
                
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
        best_model_name, (best_model, _) = sorted_models[0]
        
        if len(sorted_models) + 1 < len(axes.flat):
            ax = axes.flat[len(sorted_models) + 1]
            with torch.no_grad():
                y_pred = best_model(x_sample).squeeze()
            
            error = (y_true - y_pred).numpy()
            ax.plot(error, 'b-', linewidth=1, alpha=0.8)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title(f'Error: Best Model ({best_model_name})\nMax={np.abs(error).max():.2e}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Node Index')
            ax.set_ylabel('Error')
            ax.grid(True, alpha=0.3)
        
        # Histogram of errors
        if len(sorted_models) + 2 < len(axes.flat):
            ax = axes.flat[len(sorted_models) + 2]
            ax.hist(error, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
            ax.set_title(f'Error Distribution\nStd={np.std(error):.2e}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Error')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(sorted_models) + 3, len(axes.flat)):
            axes.flat[idx].set_visible(False)
        
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
    
    # Get only two_matrix models and sort them
    two_matrix_models = {k: v for k, v in models.items() if 'two_matrix' in k}
    sorted_models = sorted(two_matrix_models.items(), key=lambda x: x[1][1])  # Sort by val_loss
    
    # Determine grid size
    n_models = len(two_matrix_models)
    n_cols = 3
    n_rows = max(2, (n_models + 3) // n_cols)  # +3 for input, ground truth, error
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Input field
    ax = axes.flat[0]
    cf = ax.tricontourf(triang, input_vec, levels=20, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_title(f'Input: sin(2π·{k1}·x)×sin(2π·{k2}·y)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Ground truth
    ax = axes.flat[1]
    cf = ax.tricontourf(triang, y_true.numpy(), levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_title('Ground Truth K@f', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Model predictions
    for idx, (model_name, (model, val_loss)) in enumerate(sorted_models):
        if idx + 2 < len(axes.flat):
            ax = axes.flat[idx + 2]
            
            with torch.no_grad():
                y_pred = model(x_sample).squeeze().numpy()
            
            cf = ax.tricontourf(triang, y_pred, levels=20, cmap='viridis')
            plt.colorbar(cf, ax=ax, fraction=0.046)
            
            mse = np.mean((y_true.numpy() - y_pred) ** 2)
            ax.set_title(f'{model_name}\nMSE={mse:.2e}', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
    
    # Error field for best model
    best_model_name, (best_model, _) = sorted_models[0]
    
    if len(sorted_models) + 2 < len(axes.flat):
        ax = axes.flat[len(sorted_models) + 2]
        with torch.no_grad():
            y_pred = best_model(x_sample).squeeze().numpy()
        
        error = np.abs(y_true.numpy() - y_pred)
        cf = ax.tricontourf(triang, error, levels=20, cmap='Reds')
        plt.colorbar(cf, ax=ax, fraction=0.046)
        ax.set_title(f'Absolute Error ({best_model_name})\nMax={error.max():.2e}', 
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(len(sorted_models) + 3, len(axes.flat)):
        axes.flat[idx].set_visible(False)
    
    plt.suptitle(f'2D Field Comparison - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'dense_2d_fields_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved 2D field comparison: {save_path}")
    plt.close()


def create_ood_tests_with_spmv(models, save_dir):
    """Test models on out-of-distribution inputs and compare with actual SpMV."""
    
    print("\n" + "="*60)
    print("OUT-OF-DISTRIBUTION TESTING WITH ACTUAL SpMV")
    print("="*60)
    
    # Load the sparse matrix K
    print("\nLoading sparse matrix K...")
    K_sparse = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    print(f"Sparse matrix K shape: {K_sparse.shape}")
    print(f"Number of non-zeros: {K_sparse.nnz}")
    
    # Get only two_matrix models
    two_matrix_models = {k: v for k, v in models.items() if 'two_matrix' in k}
    sorted_models = sorted(two_matrix_models.items(), key=lambda x: x[1][1])
    
    n_nodes = 648  # Number of nodes
    n_samples = 10  # Number of OOD samples to test
    
    # Create different OOD test cases
    ood_cases = {}
    
    # 1. Gaussian random field (smooth)
    print("\nGenerating OOD test cases...")
    np.random.seed(123)
    gaussian_smooth = []
    for _ in range(n_samples):
        # Generate smooth Gaussian random field using low-frequency components
        freqs = np.random.randn(10) * 0.5  # Few frequency components
        x = np.zeros(n_nodes)
        for i, freq in enumerate(freqs):
            x += freq * np.sin(2 * np.pi * (i+1) * np.linspace(0, 1, n_nodes))
        x = x / (np.std(x) + 1e-8)  # Normalize
        gaussian_smooth.append(x)
    ood_cases['Gaussian Smooth'] = gaussian_smooth
    
    # 2. Pure random noise
    random_noise = []
    for _ in range(n_samples):
        x = np.random.randn(n_nodes)
        random_noise.append(x)
    ood_cases['Random Noise'] = random_noise
    
    # 3. Step functions (discontinuous)
    step_functions = []
    for i in range(n_samples):
        x = np.zeros(n_nodes)
        n_steps = np.random.randint(2, 6)
        step_positions = sorted(np.random.choice(n_nodes, n_steps, replace=False))
        for j in range(len(step_positions)-1):
            x[step_positions[j]:step_positions[j+1]] = np.random.randn()
        step_functions.append(x)
    ood_cases['Step Functions'] = step_functions
    
    # 4. High-frequency oscillations
    high_freq = []
    for _ in range(n_samples):
        freq = np.random.uniform(20, 50)
        phase = np.random.uniform(0, 2*np.pi)
        x = np.sin(freq * np.linspace(0, 2*np.pi, n_nodes) + phase)
        x += 0.2 * np.random.randn(n_nodes)  # Add some noise
        high_freq.append(x)
    ood_cases['High Frequency'] = high_freq
    
    # 5. Exponential decay patterns
    exp_patterns = []
    for _ in range(n_samples):
        decay_rate = np.random.uniform(0.01, 0.1)
        x = np.exp(-decay_rate * np.arange(n_nodes))
        x = x * np.random.choice([-1, 1])  # Random sign
        x += 0.1 * np.random.randn(n_nodes)  # Add noise
        exp_patterns.append(x)
    ood_cases['Exponential'] = exp_patterns
    
    # Compute actual SpMV results and compare with model predictions
    results = {}
    
    for case_name, inputs in ood_cases.items():
        print(f"\nAnalyzing {case_name}:")
        case_results = {}
        
        # Compute actual SpMV results
        true_outputs = []
        for x in inputs:
            y_true = K_sparse.dot(x)
            true_outputs.append(y_true)
        true_outputs = np.array(true_outputs)
        
        print(f"  True SpMV output range: [{true_outputs.min():.3f}, {true_outputs.max():.3f}]")
        print(f"  True SpMV output std: {true_outputs.std():.3f}")
        
        # Test each model
        for model_name, (model, _) in sorted_models:
            outputs = []
            errors = []
            relative_errors = []
            
            with torch.no_grad():
                for i, x in enumerate(inputs):
                    x_tensor = torch.FloatTensor(x)
                    y_pred = model(x_tensor.unsqueeze(0)).squeeze().numpy()
                    outputs.append(y_pred)
                    
                    # Calculate error vs true SpMV
                    error = y_pred - true_outputs[i]
                    errors.append(error)
                    
                    # Calculate relative error
                    rel_error = np.abs(error) / (np.abs(true_outputs[i]) + 1e-10)
                    relative_errors.append(rel_error.mean())
            
            outputs = np.array(outputs)
            errors = np.array(errors)
            
            # Calculate statistics
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))
            mean_rel_error = np.mean(relative_errors)
            
            case_results[model_name] = {
                'outputs': outputs,
                'true_outputs': true_outputs,
                'errors': errors,
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'relative_error': mean_rel_error
            }
            
            print(f"  {model_name}:")
            print(f"    MSE vs true SpMV: {mse:.6f}")
            print(f"    MAE vs true SpMV: {mae:.6f}")
            print(f"    Max error: {max_error:.6f}")
            print(f"    Mean relative error: {mean_rel_error:.3%}")
        
        results[case_name] = case_results
    
    # Create comprehensive visualization
    create_ood_spmv_visualization(ood_cases, results, sorted_models, save_dir)
    
    return results


def create_ood_spmv_visualization(ood_cases, results, sorted_models, save_dir):
    """Create visualizations comparing OOD predictions with actual SpMV."""
    
    n_cases = len(ood_cases)
    n_models = len(sorted_models)
    
    # Create error comparison figure
    fig, axes = plt.subplots(n_cases, n_models + 1, figsize=(5*(n_models+1), 4*n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    for case_idx, (case_name, inputs) in enumerate(ood_cases.items()):
        # Select one representative sample
        sample_idx = 0
        sample_input = inputs[sample_idx]
        
        # Plot input and true SpMV output
        ax = axes[case_idx, 0]
        true_output = results[case_name][sorted_models[0][0]]['true_outputs'][sample_idx]
        
        ax.plot(sample_input, 'b-', linewidth=1, alpha=0.7, label='Input')
        ax.plot(true_output, 'k-', linewidth=1.5, label='True SpMV')
        ax.set_title(f'{case_name}\nInput & True SpMV', fontsize=10, fontweight='bold')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot each model's prediction vs true
        for model_idx, (model_name, _) in enumerate(sorted_models):
            ax = axes[case_idx, model_idx + 1]
            
            model_output = results[case_name][model_name]['outputs'][sample_idx]
            error = results[case_name][model_name]['errors'][sample_idx]
            
            ax.plot(true_output, 'k-', linewidth=1, alpha=0.5, label='True SpMV')
            ax.plot(model_output, 'g--', linewidth=1, alpha=0.7, label='Model')
            
            # Add error subplot
            ax2 = ax.twinx()
            ax2.plot(np.abs(error), 'r-', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel('|Error|', color='r', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
            
            mse = results[case_name][model_name]['mse']
            mae = results[case_name][model_name]['mae']
            
            ax.set_title(f'{model_name.replace("two_matrix_", "")}\nMSE={mse:.2e}', fontsize=10)
            ax.set_xlabel('Node Index')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('OOD Predictions vs Actual SpMV', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'ood_spmv_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ OOD SpMV comparison saved to: {save_path}")
    plt.close()
    
    # Create error statistics summary
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    case_names = list(ood_cases.keys())
    model_names_short = [m.replace('two_matrix_', '') for m, _ in sorted_models]
    
    # MSE comparison
    ax = axes[0, 0]
    mse_data = np.zeros((len(case_names), len(sorted_models)))
    for i, case in enumerate(case_names):
        for j, (model, _) in enumerate(sorted_models):
            mse_data[i, j] = results[case][model]['mse']
    
    im = ax.imshow(mse_data, aspect='auto', cmap='hot')
    ax.set_xticks(range(len(model_names_short)))
    ax.set_xticklabels(model_names_short)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names)
    ax.set_title('MSE vs True SpMV', fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # MAE comparison
    ax = axes[0, 1]
    mae_data = np.zeros((len(case_names), len(sorted_models)))
    for i, case in enumerate(case_names):
        for j, (model, _) in enumerate(sorted_models):
            mae_data[i, j] = results[case][model]['mae']
    
    im = ax.imshow(mae_data, aspect='auto', cmap='hot')
    ax.set_xticks(range(len(model_names_short)))
    ax.set_xticklabels(model_names_short)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names)
    ax.set_title('MAE vs True SpMV', fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Relative error comparison
    ax = axes[0, 2]
    rel_error_data = np.zeros((len(case_names), len(sorted_models)))
    for i, case in enumerate(case_names):
        for j, (model, _) in enumerate(sorted_models):
            rel_error_data[i, j] = results[case][model]['relative_error']
    
    im = ax.imshow(rel_error_data, aspect='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_xticks(range(len(model_names_short)))
    ax.set_xticklabels(model_names_short)
    ax.set_yticks(range(len(case_names)))
    ax.set_yticklabels(case_names)
    ax.set_title('Mean Relative Error', fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Bar plot comparison for each case
    ax = axes[1, 0]
    width = 0.8 / len(sorted_models)
    x = np.arange(len(case_names))
    
    for j, (model, _) in enumerate(sorted_models):
        mse_values = [results[case][model]['mse'] for case in case_names]
        ax.bar(x + j*width, mse_values, width, label=model.replace('two_matrix_', ''))
    
    ax.set_xlabel('OOD Case')
    ax.set_ylabel('MSE')
    ax.set_title('MSE by OOD Case', fontweight='bold')
    ax.set_xticks(x + width * (len(sorted_models)-1) / 2)
    ax.set_xticklabels(case_names, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Average performance across all cases
    ax = axes[1, 1]
    avg_mse = []
    avg_mae = []
    avg_rel_error = []
    
    for model, _ in sorted_models:
        mse_vals = [results[case][model]['mse'] for case in case_names]
        mae_vals = [results[case][model]['mae'] for case in case_names]
        rel_vals = [results[case][model]['relative_error'] for case in case_names]
        
        avg_mse.append(np.mean(mse_vals))
        avg_mae.append(np.mean(mae_vals))
        avg_rel_error.append(np.mean(rel_vals))
    
    x = np.arange(len(model_names_short))
    width = 0.25
    
    ax.bar(x - width, avg_mse, width, label='MSE', color='red', alpha=0.7)
    ax.bar(x, avg_mae, width, label='MAE', color='green', alpha=0.7)
    ax.bar(x + width, avg_rel_error, width, label='Rel Error', color='blue', alpha=0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Error')
    ax.set_title('Average OOD Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_short)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 2]
    for model_idx, (model, _) in enumerate(sorted_models):
        all_errors = []
        for case in case_names:
            errors = results[case][model]['errors'].flatten()
            all_errors.extend(errors)
        
        ax.hist(all_errors, bins=50, alpha=0.5, label=model.replace('two_matrix_', ''), 
                density=True)
    
    ax.set_xlabel('Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution (All OOD Cases)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('OOD Performance Summary vs Actual SpMV', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'ood_error_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ OOD error summary saved to: {save_path}")
    plt.close()


def investigate_perfect_predictions(models, X_val, Y_val, save_dir):
    """Investigate why predictions are suspiciously perfect."""
    
    print("\n" + "="*60)
    print("INVESTIGATING SUSPICIOUSLY PERFECT PREDICTIONS")
    print("="*60)
    
    # Get the best two_matrix model
    two_matrix_models = {k: v for k, v in models.items() if 'two_matrix' in k}
    if not two_matrix_models:
        print("No two_matrix models found!")
        return None, None, None
    
    best_model_name = min(two_matrix_models.keys(), key=lambda k: two_matrix_models[k][1])
    model, val_loss = two_matrix_models[best_model_name]
    
    # Check weight matrices
    W1 = model.W1.detach().numpy()
    W2 = model.W2.detach().numpy()
    bias = model.bias.detach().numpy()
    
    print(f"\nModel: {best_model_name}")
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
    axes[0].set_title(f'W1 Matrix ({W1.shape[0]}×{W1.shape[1]})\nCond={cond_W1:.2e}')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])
    
    # W2 matrix (show subset for visibility)
    if W2.shape[0] > 50:
        W2_subset = W2[:50, :50] if W2.shape[1] > 50 else W2[:50, :]
        title = f'W2 Matrix (first 50×{min(50, W2.shape[1])} of {W2.shape[0]}×{W2.shape[1]})'
    else:
        W2_subset = W2
        title = f'W2 Matrix ({W2.shape[0]}×{W2.shape[1]})'
    
    im2 = axes[1].imshow(W2_subset, cmap='RdBu_r', aspect='auto')
    axes[1].set_title(title)
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
    
    # Test on out-of-distribution data with actual SpMV comparison
    ood_results = create_ood_tests_with_spmv(models, save_dir)
    
    # Investigate perfect predictions
    W1, W2, bias = investigate_perfect_predictions(models, X_val, Y_val, save_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n--- In-Distribution Performance ---")
    for model_name in all_metrics:
        metrics = all_metrics[model_name]
        print(f"\n{model_name}:")
        print(f"  Mean MSE: {np.mean(metrics['mse']):.6e}")
        print(f"  Mean MAE: {np.mean(metrics['mae']):.6e}")
        print(f"  Mean Max Error: {np.mean(metrics['max_error']):.6e}")
    
    print("\n--- Out-of-Distribution Performance vs Actual SpMV ---")
    if ood_results:
        # Calculate average OOD performance for each model
        two_matrix_models = [k for k in models.keys() if 'two_matrix' in k]
        for model_name in sorted(two_matrix_models):
            ood_mse = []
            ood_mae = []
            ood_rel_error = []
            
            for case_name in ood_results:
                if model_name in ood_results[case_name]:
                    ood_mse.append(ood_results[case_name][model_name]['mse'])
                    ood_mae.append(ood_results[case_name][model_name]['mae'])
                    ood_rel_error.append(ood_results[case_name][model_name]['relative_error'])
            
            if ood_mse:
                print(f"\n{model_name} (OOD average):")
                print(f"  MSE vs true SpMV: {np.mean(ood_mse):.6f}")
                print(f"  MAE vs true SpMV: {np.mean(ood_mae):.6f}")
                print(f"  Mean relative error: {np.mean(ood_rel_error):.3%}")
    
    print(f"\n✅ All visualizations saved to: {save_dir}")
    print("\n⚠️ Check the plots carefully for signs of overfitting or memorization!")
    
    # Save metrics
    import json
    with open(os.path.join(save_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump({k: {kk: float(np.mean(vv)) for kk, vv in v.items()} 
                  for k, v in all_metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
