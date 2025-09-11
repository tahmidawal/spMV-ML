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
from train_dense_spmv import DenseSpMV_Simple, DenseSpMV_TwoMatrix, DenseSpMV_LowRank


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
    
    # Load two_matrix 2x324 (best performer)
    checkpoint = torch.load(os.path.join(latest_dir, 'two_matrix_2x324_best.pth'), map_location='cpu')
    model_2x324 = DenseSpMV_TwoMatrix(vector_dim=648, matrix_shape=(2, 324))
    model_2x324.load_state_dict(checkpoint['model_state_dict'])
    model_2x324.eval()
    models['two_matrix_2x324'] = (model_2x324, checkpoint['val_loss'])
    
    # Load two_matrix 3x216
    checkpoint = torch.load(os.path.join(latest_dir, 'two_matrix_3x216_best.pth'), map_location='cpu')
    model_3x216 = DenseSpMV_TwoMatrix(vector_dim=648, matrix_shape=(3, 216))
    model_3x216.load_state_dict(checkpoint['model_state_dict'])
    model_3x216.eval()
    models['two_matrix_3x216'] = (model_3x216, checkpoint['val_loss'])
    
    # Load simple model for comparison
    checkpoint = torch.load(os.path.join(latest_dir, 'simple_3x216_best.pth'), map_location='cpu')
    model_simple = DenseSpMV_Simple(vector_dim=648, matrix_shape=(3, 216))
    model_simple.load_state_dict(checkpoint['model_state_dict'])
    model_simple.eval()
    models['simple_3x216'] = (model_simple, checkpoint['val_loss'])
    
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
    
    # Investigate perfect predictions
    W1, W2, bias = investigate_perfect_predictions(models, X_val, Y_val, save_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_name in all_metrics:
        metrics = all_metrics[model_name]
        print(f"\n{model_name}:")
        print(f"  Mean MSE: {np.mean(metrics['mse']):.6e}")
        print(f"  Mean MAE: {np.mean(metrics['mae']):.6e}")
        print(f"  Mean Max Error: {np.mean(metrics['max_error']):.6e}")
    
    print(f"\n✅ All visualizations saved to: {save_dir}")
    print("\n⚠️ Check the plots carefully for signs of overfitting or memorization!")
    
    # Save metrics
    import json
    with open(os.path.join(save_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump({k: {kk: float(np.mean(vv)) for kk, vv in v.items()} 
                  for k, v in all_metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
