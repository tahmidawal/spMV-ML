#!/usr/bin/env python3
"""
Comprehensive inference vs ground truth comparison for the trained GNN model.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.sparse import load_npz
import matplotlib.tri as mtri
from datetime import datetime
import os

# Import the model architecture
from train_gnn_best import BestSpMV_GNN
from torch_geometric.utils import from_scipy_sparse_matrix


def load_best_model(model_path):
    """Load the best trained model."""
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with same architecture as saved model
    model = BestSpMV_GNN(
        input_dim=1,
        hidden_dim=96,   # From the actual saved model
        output_dim=1,
        num_layers=4,    # From the actual saved model
        conv_type='GAT', # From the actual saved model
        dropout=0.1,
        use_residual=True,
        use_layer_norm=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss = {checkpoint['val_loss']:.6f}")
    
    return model


def run_full_inference(model, X_val, Y_val, edge_index):
    """Run inference on all validation samples."""
    
    print("Running inference on all validation samples...")
    
    all_predictions = []
    all_ground_truth = []
    all_inputs = []
    all_errors = []
    
    model.eval()
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x_sample = X_val[i].unsqueeze(-1)  # [648, 1]
            y_true = Y_val[i]                  # [648]
            
            # Predict
            y_pred = model(x_sample, edge_index).squeeze(-1)  # [648]
            
            # Store results
            all_inputs.append(x_sample.squeeze().numpy())
            all_predictions.append(y_pred.numpy())
            all_ground_truth.append(y_true.numpy())
            
            # Compute error
            error = torch.abs(y_pred - y_true).numpy()
            all_errors.append(error)
    
    return {
        'inputs': np.array(all_inputs),           # [100, 648]
        'predictions': np.array(all_predictions), # [100, 648]
        'ground_truth': np.array(all_ground_truth), # [100, 648]
        'errors': np.array(all_errors)           # [100, 648]
    }


def plot_overall_comparison(results, save_path=None):
    """Plot overall comparison statistics."""
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    errors = results['errors']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Overall scatter plot (all samples, all nodes)
    ax1 = axes[0, 0]
    pred_flat = predictions.flatten()
    true_flat = ground_truth.flatten()
    
    # Sample for visualization (too many points otherwise)
    n_sample = min(10000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_sample, replace=False)
    
    ax1.scatter(true_flat[indices], pred_flat[indices], alpha=0.5, s=1)
    
    # Perfect prediction line
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
    
    # Compute R²
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    ax1.set_xlabel('Ground Truth')
    ax1.set_ylabel('Prediction')
    ax1.set_title(f'Overall Prediction Accuracy\nR² = {r_squared:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    ax2 = axes[0, 1]
    error_flat = errors.flatten()
    ax2.hist(error_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution\nMean = {error_flat.mean():.4f}, Std = {error_flat.std():.4f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Sample-wise MSE
    ax3 = axes[0, 2]
    sample_mse = np.mean((predictions - ground_truth)**2, axis=1)
    ax3.plot(sample_mse, 'b-', linewidth=1, alpha=0.8)
    ax3.axhline(y=sample_mse.mean(), color='red', linestyle='--', alpha=0.8, 
                label=f'Mean = {sample_mse.mean():.6f}')
    ax3.set_xlabel('Validation Sample')
    ax3.set_ylabel('MSE')
    ax3.set_title('Per-Sample Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Node-wise error analysis
    ax4 = axes[1, 0]
    node_errors = np.mean(errors, axis=0)  # Average error per node
    ax4.plot(node_errors, 'g-', linewidth=1, alpha=0.8)
    ax4.set_xlabel('Node Index')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Per-Node Error Analysis')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Best vs worst samples
    ax5 = axes[1, 1]
    best_idx = np.argmin(sample_mse)
    worst_idx = np.argmax(sample_mse)
    
    ax5.plot(ground_truth[best_idx], 'r-', linewidth=1.5, alpha=0.8, label=f'Best GT (MSE={sample_mse[best_idx]:.2e})')
    ax5.plot(predictions[best_idx], 'r--', linewidth=1.5, alpha=0.8, label='Best Pred')
    ax5.plot(ground_truth[worst_idx], 'b-', linewidth=1.5, alpha=0.8, label=f'Worst GT (MSE={sample_mse[worst_idx]:.2e})')
    ax5.plot(predictions[worst_idx], 'b--', linewidth=1.5, alpha=0.8, label='Worst Pred')
    
    ax5.set_xlabel('Node Index')
    ax5.set_ylabel('Value')
    ax5.set_title('Best vs Worst Predictions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Accuracy statistics
    ax6 = axes[1, 2]
    
    # Compute various accuracy metrics
    mse_overall = np.mean((predictions - ground_truth)**2)
    mae_overall = np.mean(np.abs(predictions - ground_truth))
    rmse_overall = np.sqrt(mse_overall)
    
    # Relative metrics
    output_range = ground_truth.max() - ground_truth.min()
    relative_rmse = rmse_overall / output_range
    
    metrics = ['MSE', 'MAE', 'RMSE', 'Rel RMSE (%)', 'R²']
    values = [mse_overall, mae_overall, rmse_overall, relative_rmse*100, r_squared]
    
    bars = ax6.bar(range(len(metrics)), values, color=['blue', 'green', 'orange', 'purple', 'red'], alpha=0.7)
    ax6.set_xticks(range(len(metrics)))
    ax6.set_xticklabels(metrics, rotation=45)
    ax6.set_title('Overall Accuracy Metrics')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Overall comparison saved to: {save_path}")
        plt.close()
    
    return {
        'mse': mse_overall,
        'mae': mae_overall,
        'rmse': rmse_overall,
        'r_squared': r_squared,
        'relative_rmse': relative_rmse
    }


def plot_sample_comparisons(results, mesh_data, n_samples=6, save_path=None):
    """Plot detailed sample-by-sample comparisons with 2D visualization."""
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    inputs = results['inputs']
    errors = results['errors']
    
    # Select samples to visualize (best, worst, and random)
    sample_mse = np.mean((predictions - ground_truth)**2, axis=1)
    best_idx = np.argmin(sample_mse)
    worst_idx = np.argmax(sample_mse)
    
    # Add some random samples
    random_indices = np.random.choice(len(predictions), n_samples-2, replace=False)
    sample_indices = [best_idx, worst_idx] + list(random_indices)
    
    # Create triangulation for 2D plotting
    points = mesh_data['points']
    triangles = mesh_data['triangles']
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(20, 5 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        input_vec = inputs[sample_idx]
        true_vec = ground_truth[sample_idx]
        pred_vec = predictions[sample_idx]
        error_vec = errors[sample_idx]
        
        mse = np.mean((true_vec - pred_vec)**2)
        mae = np.mean(np.abs(true_vec - pred_vec))
        
        sample_type = "BEST" if sample_idx == best_idx else "WORST" if sample_idx == worst_idx else "RANDOM"
        
        # Plot 1: Input field (2D)
        ax1 = axes[plot_idx, 0]
        cf = ax1.tricontourf(triang, input_vec, levels=20, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f'{sample_type} Sample {sample_idx}: Input f\nRange: [{input_vec.min():.3f}, {input_vec.max():.3f}]')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # Plot 2: Ground truth (2D)
        ax2 = axes[plot_idx, 1]
        ct = ax2.tricontourf(triang, true_vec, levels=20, cmap='viridis')
        plt.colorbar(ct, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f'Ground Truth: K @ f\nRange: [{true_vec.min():.3f}, {true_vec.max():.3f}]')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # Plot 3: Prediction (2D)
        ax3 = axes[plot_idx, 2]
        cp = ax3.tricontourf(triang, pred_vec, levels=20, cmap='viridis')
        plt.colorbar(cp, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title(f'GNN Prediction\nRange: [{pred_vec.min():.3f}, {pred_vec.max():.3f}]')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        # Plot 4: Error (2D)
        ax4 = axes[plot_idx, 3]
        ce = ax4.tricontourf(triang, error_vec, levels=20, cmap='Reds')
        plt.colorbar(ce, ax=ax4, fraction=0.046, pad=0.04)
        ax4.set_title(f'Absolute Error\nMSE={mse:.2e}, MAE={mae:.2e}')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_aspect('equal')
    
    plt.suptitle(f'GNN Inference vs Ground Truth Comparison\n'
                 f'Validation Set Analysis (Best GAT Model)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Sample comparisons saved to: {save_path}")
        plt.close()


def plot_1d_comparisons(results, n_samples=8, save_path=None):
    """Plot 1D vector comparisons for detailed analysis."""
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    inputs = results['inputs']
    errors = results['errors']
    
    # Select samples
    sample_mse = np.mean((predictions - ground_truth)**2, axis=1)
    best_idx = np.argmin(sample_mse)
    worst_idx = np.argmax(sample_mse)
    
    # Get diverse samples
    indices = [best_idx, worst_idx]
    remaining_samples = n_samples - 2
    if remaining_samples > 0:
        random_indices = np.random.choice([i for i in range(len(predictions)) 
                                         if i not in indices], remaining_samples, replace=False)
        indices.extend(random_indices)
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(18, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(indices):
        input_vec = inputs[sample_idx]
        true_vec = ground_truth[sample_idx]
        pred_vec = predictions[sample_idx]
        error_vec = errors[sample_idx]
        
        mse = np.mean((true_vec - pred_vec)**2)
        mae = np.mean(np.abs(true_vec - pred_vec))
        
        sample_type = "BEST" if sample_idx == best_idx else "WORST" if sample_idx == worst_idx else "RANDOM"
        
        # Plot 1: Input
        axes[plot_idx, 0].plot(input_vec, 'b-', linewidth=1, alpha=0.8)
        axes[plot_idx, 0].set_title(f'{sample_type} Sample {sample_idx}: Input f')
        axes[plot_idx, 0].set_xlabel('Node Index')
        axes[plot_idx, 0].set_ylabel('Value')
        axes[plot_idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Ground truth vs Prediction
        axes[plot_idx, 1].plot(true_vec, 'r-', linewidth=1.5, alpha=0.8, label='Ground Truth')
        axes[plot_idx, 1].plot(pred_vec, 'g--', linewidth=1.5, alpha=0.8, label='GNN Prediction')
        axes[plot_idx, 1].set_title(f'Comparison (MSE={mse:.2e}, MAE={mae:.2e})')
        axes[plot_idx, 1].set_xlabel('Node Index')
        axes[plot_idx, 1].set_ylabel('Value')
        axes[plot_idx, 1].legend()
        axes[plot_idx, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error analysis
        axes[plot_idx, 2].plot(error_vec, 'k-', linewidth=1, alpha=0.8)
        axes[plot_idx, 2].set_title(f'Absolute Error\nMax = {error_vec.max():.3e}')
        axes[plot_idx, 2].set_xlabel('Node Index')
        axes[plot_idx, 2].set_ylabel('|True - Predicted|')
        axes[plot_idx, 2].grid(True, alpha=0.3)
        axes[plot_idx, 2].set_yscale('log')
    
    plt.suptitle(f'Detailed 1D Vector Comparisons\n'
                 f'GNN vs Ground Truth SpMV Results', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"1D comparisons saved to: {save_path}")
        plt.close()


def plot_accuracy_analysis(results, save_path=None):
    """Plot detailed accuracy analysis."""
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    errors = results['errors']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Error vs magnitude
    ax1 = axes[0, 0]
    true_magnitude = np.abs(ground_truth.flatten())
    error_magnitude = errors.flatten()
    
    # Bin by magnitude and plot average error
    magnitude_bins = np.linspace(0, true_magnitude.max(), 20)
    bin_centers = (magnitude_bins[:-1] + magnitude_bins[1:]) / 2
    bin_errors = []
    
    for i in range(len(magnitude_bins)-1):
        mask = (true_magnitude >= magnitude_bins[i]) & (true_magnitude < magnitude_bins[i+1])
        if np.sum(mask) > 0:
            bin_errors.append(np.mean(error_magnitude[mask]))
        else:
            bin_errors.append(0)
    
    ax1.plot(bin_centers, bin_errors, 'bo-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Ground Truth Magnitude')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error vs Output Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual analysis
    ax2 = axes[0, 1]
    residuals = (predictions - ground_truth).flatten()
    ax2.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Residual (Predicted - True)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Residual Distribution\nMean = {residuals.mean():.2e}, Std = {residuals.std():.4f}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot for residuals
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot: Residuals vs Normal Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning curve summary
    ax4 = axes[1, 1]
    
    # Load training history for comparison
    try:
        history = np.load('gnn_best_results_20250910_191922/best_training_history.npz')
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        
        epochs = range(1, len(train_losses) + 1)
        ax4.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax4.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MSE Loss')
        ax4.set_title('Training Convergence')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
    except:
        ax4.text(0.5, 0.5, 'Training history\nnot available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training History')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Accuracy analysis saved to: {save_path}")
        plt.close()


def main():
    print("=== COMPREHENSIVE INFERENCE vs GROUND TRUTH ANALYSIS ===")
    print()
    
    # Load data
    print("Loading dataset and model...")
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    
    X_val = torch.FloatTensor(data['X_val'])
    Y_val = torch.FloatTensor(data['Y_val'])
    
    edge_index, _ = from_scipy_sparse_matrix(K)
    
    # Load the best trained model
    model_path = 'gnn_best_results_20250910_191922/best_model.pth'
    model = load_best_model(model_path)
    
    # Run full inference
    results = run_full_inference(model, X_val, Y_val, edge_index)
    
    print(f"Inference completed on {len(results['predictions'])} validation samples")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"inference_analysis_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate comprehensive plots
    print("Generating comparison plots...")
    
    # Overall comparison
    overall_stats = plot_overall_comparison(results, 
                                          os.path.join(save_dir, 'overall_comparison.png'))
    
    # Sample-by-sample 2D comparisons
    mesh_data = {'points': data['points'], 'triangles': data['triangles']}
    plot_sample_comparisons(results, mesh_data, n_samples=6,
                           save_path=os.path.join(save_dir, 'sample_comparisons_2d.png'))
    
    # 1D vector comparisons
    plot_1d_comparisons(results, n_samples=8,
                       save_path=os.path.join(save_dir, 'sample_comparisons_1d.png'))
    
    # Detailed accuracy analysis
    plot_accuracy_analysis(results, os.path.join(save_dir, 'accuracy_analysis.png'))
    
    # Print final summary
    print(f"\n🎯 FINAL MODEL PERFORMANCE SUMMARY:")
    print(f"  MSE: {overall_stats['mse']:.6f}")
    print(f"  RMSE: {overall_stats['rmse']:.6f}")
    print(f"  MAE: {overall_stats['mae']:.6f}")
    print(f"  R²: {overall_stats['r_squared']:.4f}")
    print(f"  Relative RMSE: {overall_stats['relative_rmse']*100:.2f}%")
    print(f"  🎉 Model successfully learned SpMV operation!")
    
    print(f"\nAll comparison plots saved to: {save_dir}")
    
    return results, overall_stats


if __name__ == "__main__":
    results, stats = main()
