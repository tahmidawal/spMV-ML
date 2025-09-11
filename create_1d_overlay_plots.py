#!/usr/bin/env python3
"""
Create 1D vector overlay plots with ground truth and predictions on the same plot.
High-quality vector formats for detailed examination.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.tri as mtri
from scipy.sparse import load_npz
from datetime import datetime
import os

# Import the model architecture
from train_gnn_best import BestSpMV_GNN
from torch_geometric.utils import from_scipy_sparse_matrix


def load_model_and_data():
    """Load the trained model and validation data."""
    
    # Load data
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250910_161619.npz')
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    
    X_val = torch.FloatTensor(data['X_val'])
    Y_val = torch.FloatTensor(data['Y_val'])
    ks_val = data['ks_val']
    
    edge_index, _ = from_scipy_sparse_matrix(K)
    
    # Load trained model
    checkpoint = torch.load('gnn_best_results_20250911_111202/best_model.pth', map_location='cpu')
    
    model = BestSpMV_GNN(
        input_dim=1,
        hidden_dim=96,
        output_dim=1,
        num_layers=4,
        conv_type='GAT',
        dropout=0.1,
        use_residual=True,
        use_layer_norm=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded GAT model with val_loss = {checkpoint['val_loss']:.6f}")
    
    return model, X_val, Y_val, ks_val, edge_index, data


def create_2d_field_plots(true_vec, pred_vec, input_vec, triang, sample_idx, k1, k2, 
                         mse, mae, save_dir):
    """Create individual 2D field plots: input, ground truth, and prediction."""
    
    # Plot 1: Input field
    plt.figure(figsize=(12, 10))
    
    cf_input = plt.tricontourf(triang, input_vec, levels=25, cmap='RdBu_r', extend='both')
    contour_input = plt.tricontour(triang, input_vec, levels=10, colors='black', 
                                  linewidths=0.6, alpha=0.8)
    plt.clabel(contour_input, inline=True, fontsize=9, fmt='%.3f')
    
    # Highlight boundary zeros
    boundary_mask = np.abs(input_vec) < 1e-10
    if np.any(boundary_mask):
        boundary_points_x = triang.x[boundary_mask]
        boundary_points_y = triang.y[boundary_mask]
        plt.scatter(boundary_points_x, boundary_points_y, 
                   c='white', s=25, edgecolors='black', linewidths=1, 
                   marker='s', alpha=0.9, label=f'Boundary Zeros ({np.sum(boundary_mask)})')
        plt.legend(fontsize=11)
    
    cbar_input = plt.colorbar(cf_input, fraction=0.046, pad=0.04)
    cbar_input.set_label('Input Field Value f(x,y)', fontsize=12)
    
    plt.title(f'2D Input Field: f(x,y) = sin(2π·{k1}·x) × sin(2π·{k2}·y)\n'
              f'Sample {sample_idx} | Range: [{input_vec.min():.4f}, {input_vec.max():.4f}]', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    # Save input field
    base_name = f'sample_{sample_idx:03d}_2d_input_k1{k1}_k2{k2}'
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"  2D input field saved: {base_name}.[pdf/svg/png]")
    plt.close()
    
    # Plot 2: Ground truth output
    plt.figure(figsize=(12, 10))
    
    cf_true = plt.tricontourf(triang, true_vec, levels=25, cmap='viridis', extend='both')
    contour_true = plt.tricontour(triang, true_vec, levels=10, colors='white', 
                                 linewidths=0.6, alpha=0.8)
    plt.clabel(contour_true, inline=True, fontsize=9, fmt='%.3f')
    
    cbar_true = plt.colorbar(cf_true, fraction=0.046, pad=0.04)
    cbar_true.set_label('Ground Truth K@f Value', fontsize=12)
    
    plt.title(f'2D Ground Truth Output: K @ f\n'
              f'Sample {sample_idx} | Range: [{true_vec.min():.4f}, {true_vec.max():.4f}]', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    # Save ground truth
    base_name = f'sample_{sample_idx:03d}_2d_ground_truth_k1{k1}_k2{k2}'
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"  2D ground truth saved: {base_name}.[pdf/svg/png]")
    plt.close()
    
    # Plot 3: Prediction output
    plt.figure(figsize=(12, 10))
    
    cf_pred = plt.tricontourf(triang, pred_vec, levels=25, cmap='viridis', extend='both')
    contour_pred = plt.tricontour(triang, pred_vec, levels=10, colors='white', 
                                 linewidths=0.6, alpha=0.8)
    plt.clabel(contour_pred, inline=True, fontsize=9, fmt='%.3f')
    
    cbar_pred = plt.colorbar(cf_pred, fraction=0.046, pad=0.04)
    cbar_pred.set_label('GNN Predicted Value', fontsize=12)
    
    # Add accuracy info
    stats_text = f'Accuracy:\nMSE = {mse:.2e}\nMAE = {mae:.2e}\nRMSE = {np.sqrt(mse):.2e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95))
    
    plt.title(f'2D GNN Prediction Output\n'
              f'Sample {sample_idx} | Range: [{pred_vec.min():.4f}, {pred_vec.max():.4f}]', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    # Save prediction
    base_name = f'sample_{sample_idx:03d}_2d_prediction_k1{k1}_k2{k2}'
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"  2D prediction saved: {base_name}.[pdf/svg/png]")
    plt.close()
    
    # Plot 4: 2D Error field
    plt.figure(figsize=(12, 10))
    
    error_2d = np.abs(true_vec - pred_vec)
    cf_error = plt.tricontourf(triang, error_2d, levels=25, cmap='Reds', extend='max')
    contour_error = plt.tricontour(triang, error_2d, levels=8, colors='darkred', 
                                  linewidths=0.6, alpha=0.8)
    plt.clabel(contour_error, inline=True, fontsize=9, fmt='%.2e')
    
    cbar_error = plt.colorbar(cf_error, fraction=0.046, pad=0.04)
    cbar_error.set_label('Absolute Error |True - Predicted|', fontsize=12)
    
    plt.title(f'2D Error Field: |Ground Truth - Prediction|\n'
              f'Sample {sample_idx} | Max Error: {error_2d.max():.2e}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    # Save error field
    base_name = f'sample_{sample_idx:03d}_2d_error_k1{k1}_k2{k2}'
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"  2D error field saved: {base_name}.[pdf/svg/png]")
    plt.close()


def create_2d_combined_plot(true_vec, pred_vec, input_vec, triang, sample_idx, k1, k2, 
                           mse, mae, save_dir):
    """Create 2D plot with input, ground truth, and prediction all on the same figure."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Find common color scale for ground truth and prediction
    vmin_output = min(true_vec.min(), pred_vec.min())
    vmax_output = max(true_vec.max(), pred_vec.max())
    
    # Plot 1: Input field
    ax1 = axes[0]
    cf_input = ax1.tricontourf(triang, input_vec, levels=20, cmap='RdBu_r', extend='both')
    contour_input = ax1.tricontour(triang, input_vec, levels=8, colors='black', 
                                  linewidths=0.5, alpha=0.7)
    ax1.clabel(contour_input, inline=True, fontsize=8, fmt='%.2f')
    
    cbar_input = plt.colorbar(cf_input, ax=ax1, fraction=0.046, pad=0.04)
    cbar_input.set_label('Input f(x,y)', fontsize=11)
    
    ax1.set_title(f'Input Field\nsin(2π·{k1}·x)×sin(2π·{k2}·y)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_aspect('equal')
    
    # Plot 2: Ground truth
    ax2 = axes[1]
    cf_true = ax2.tricontourf(triang, true_vec, levels=20, cmap='viridis', 
                             vmin=vmin_output, vmax=vmax_output, extend='both')
    contour_true = ax2.tricontour(triang, true_vec, levels=8, colors='white', 
                                 linewidths=0.5, alpha=0.7)
    ax2.clabel(contour_true, inline=True, fontsize=8, fmt='%.2f')
    
    cbar_true = plt.colorbar(cf_true, ax=ax2, fraction=0.046, pad=0.04)
    cbar_true.set_label('Ground Truth K@f', fontsize=11)
    
    ax2.set_title(f'Ground Truth\nK @ f', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_aspect('equal')
    
    # Plot 3: Prediction
    ax3 = axes[2]
    cf_pred = ax3.tricontourf(triang, pred_vec, levels=20, cmap='viridis',
                             vmin=vmin_output, vmax=vmax_output, extend='both')
    contour_pred = ax3.tricontour(triang, pred_vec, levels=8, colors='white',
                                 linewidths=0.5, alpha=0.7)
    ax3.clabel(contour_pred, inline=True, fontsize=8, fmt='%.2f')
    
    cbar_pred = plt.colorbar(cf_pred, ax=ax3, fraction=0.046, pad=0.04)
    cbar_pred.set_label('GNN Prediction', fontsize=11)
    
    ax3.set_title(f'GNN Prediction\nMSE={mse:.2e}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    ax3.set_aspect('equal')
    
    # Overall title
    plt.suptitle(f'2D Field Comparison: Sample {sample_idx} (k1={k1}, k2={k2})\n'
                 f'Input → Ground Truth → GNN Prediction', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save combined 2D plot
    base_name = f'sample_{sample_idx:03d}_2d_combined_k1{k1}_k2{k2}'
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    print(f"  2D combined plot saved: {base_name}.[pdf/svg/png]")
    plt.close()


def create_1d_overlay_plot(true_vec, pred_vec, input_vec, sample_idx, k1, k2, 
                          mse, mae, save_dir):
    """Create a detailed 1D vector overlay plot."""
    
    plt.figure(figsize=(20, 10))  # Much wider for better detail visibility
    
    node_indices = range(len(true_vec))
    error_vec = np.abs(true_vec - pred_vec)
    
    # Main overlay plot with thinner lines for cleaner appearance
    plt.plot(node_indices, true_vec, 'r-', linewidth=1.5, alpha=0.9, 
             label='Ground Truth K@f', zorder=3)
    plt.plot(node_indices, pred_vec, 'g--', linewidth=1.5, alpha=0.9, 
             label='GNN Prediction', zorder=2)
    
    # Error shading (subtle)
    plt.fill_between(node_indices, true_vec - error_vec, true_vec + error_vec, 
                    alpha=0.15, color='gray', label='Error Band (±|error|)', zorder=1)
    
    # Highlight boundary nodes (where input ≈ 0)
    boundary_nodes = np.where(np.abs(input_vec) < 1e-10)[0]
    interior_nodes = np.where(np.abs(input_vec) >= 1e-10)[0]
    
    if len(boundary_nodes) > 0:
        plt.scatter(boundary_nodes, true_vec[boundary_nodes], 
                   c='blue', s=60, alpha=0.8, marker='o', 
                   label=f'Boundary Nodes ({len(boundary_nodes)})', zorder=4)
    
    # Add vertical lines to separate regions (optional)
    if len(boundary_nodes) > 0 and len(interior_nodes) > 0:
        first_interior = interior_nodes[0] if len(interior_nodes) > 0 else 0
        plt.axvline(x=first_interior-0.5, color='black', linestyle=':', alpha=0.5, 
                   label='Boundary/Interior Split')
    
    # Enhanced title and labels
    plt.title(f'1D Vector Overlay: Ground Truth vs GNN Prediction\n'
              f'Sample {sample_idx}: f(x,y) = sin(2π·{k1}·x) × sin(2π·{k2}·y)\n'
              f'Red = Ground Truth, Green = GNN Prediction', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel('Node Index (Mesh Point)', fontsize=16)
    plt.ylabel('Field Value', fontsize=16)
    plt.legend(fontsize=14, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.4, linewidth=0.8)
    
    # Enhance tick labels and make them larger
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=11)
    
    # Comprehensive statistics box
    correlation = np.corrcoef(true_vec, pred_vec)[0, 1]
    relative_rmse = np.sqrt(mse) / (true_vec.max() - true_vec.min())
    max_error = error_vec.max()
    mean_error = error_vec.mean()
    
    stats_text = (f'Accuracy Metrics:\n'
                 f'MSE = {mse:.4e}\n'
                 f'MAE = {mae:.4e}\n'
                 f'RMSE = {np.sqrt(mse):.4e}\n'
                 f'Correlation = {correlation:.5f}\n'
                 f'Relative RMSE = {relative_rmse:.4f}\n'
                 f'Max Error = {max_error:.4e}\n'
                 f'Mean Error = {mean_error:.4e}')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95,
                     edgecolor='gray', linewidth=1.5))
    
    # Field information box
    field_info = (f'Field Information:\n'
                 f'Total Nodes: {len(true_vec)}\n'
                 f'Boundary Nodes: {len(boundary_nodes)}\n'
                 f'Interior Nodes: {len(interior_nodes)}\n'
                 f'Input Range: [{input_vec.min():.3f}, {input_vec.max():.3f}]\n'
                 f'Output Range: [{true_vec.min():.3f}, {true_vec.max():.3f}]\n'
                 f'Frequency: k1²+k2² = {k1*k1 + k2*k2}')
    
    plt.text(0.98, 0.98, field_info, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9,
                     edgecolor='blue', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save in multiple vector formats
    base_name = f'sample_{sample_idx:03d}_1d_overlay_k1{k1}_k2{k2}'
    
    # PDF (best for detailed examination)
    pdf_path = os.path.join(save_dir, f'{base_name}.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    
    # SVG (web-compatible vector)
    svg_path = os.path.join(save_dir, f'{base_name}.svg')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    
    # EPS (publication quality vector)
    eps_path = os.path.join(save_dir, f'{base_name}.eps')
    plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    
    # PNG (high-resolution raster)
    png_path = os.path.join(save_dir, f'{base_name}.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"  1D overlay plot saved: {base_name}.[pdf/svg/eps/png]")
    
    plt.close()
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'max_error': max_error,
        'relative_rmse': relative_rmse
    }


def create_detailed_error_analysis_plot(true_vec, pred_vec, input_vec, sample_idx, k1, k2, 
                                       mse, mae, save_dir):
    """Create detailed error analysis plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))  # Wider for better detail
    
    node_indices = range(len(true_vec))
    error_vec = np.abs(true_vec - pred_vec)
    relative_error_vec = error_vec / (np.abs(true_vec) + 1e-10)  # Avoid division by zero
    
    # Plot 1: Absolute error
    axes[0, 0].plot(node_indices, error_vec, 'k-', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_title(f'Absolute Error: |Ground Truth - Prediction|', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Node Index', fontsize=13)
    axes[0, 0].set_ylabel('|Error|', fontsize=13)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].set_yscale('log')
    axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Relative error
    axes[0, 1].plot(node_indices, relative_error_vec, 'purple', linewidth=1.2, alpha=0.8)
    axes[0, 1].set_title(f'Relative Error: |Error| / |Ground Truth|', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Node Index')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Error vs ground truth magnitude
    axes[1, 0].scatter(np.abs(true_vec), error_vec, alpha=0.6, s=15)
    axes[1, 0].set_xlabel('|Ground Truth|')
    axes[1, 0].set_ylabel('|Error|')
    axes[1, 0].set_title('Error vs Ground Truth Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Residual histogram
    residuals = pred_vec - true_vec
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Residual (Predicted - True)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall statistics
    plt.suptitle(f'Error Analysis: Sample {sample_idx} (k1={k1}, k2={k2})\n'
                 f'MSE={mse:.2e}, MAE={mae:.2e}, Max Error={error_vec.max():.2e}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in vector formats
    base_name = f'sample_{sample_idx:03d}_error_analysis_k1{k1}_k2{k2}'
    
    for fmt in ['pdf', 'svg', 'eps', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    
    print(f"  Error analysis plot saved: {base_name}.[pdf/svg/eps/png]")
    plt.close()


def main():
    print("=== CREATING 1D VECTOR OVERLAY PLOTS ===")
    print()
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"1d_overlay_plots_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created directory: {save_dir}")
    
    # Load model and data
    model, X_val, Y_val, ks_val, edge_index, data = load_model_and_data()
    
    # Create triangulation for 2D plotting
    points = data['points']
    triangles = data['triangles']
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    
    # Select the same 5 samples for consistency
    np.random.seed(42)
    n_samples = 5
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    print(f"Selected samples: {sample_indices}")
    print()
    
    # Run inference and create overlay plots
    model.eval()
    all_stats = []
    
    print("Creating comprehensive 2D field and 1D vector overlay plots...")
    
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            print(f"\nProcessing Sample {sample_idx} ({i+1}/{n_samples}):")
            
            x_sample = X_val[sample_idx].unsqueeze(-1)  # [648, 1]
            y_true = Y_val[sample_idx]                  # [648]
            
            # Predict
            y_pred = model(x_sample, edge_index).squeeze(-1)  # [648]
            
            # Convert to numpy
            input_vec = x_sample.squeeze().numpy()
            true_vec = y_true.numpy()
            pred_vec = y_pred.numpy()
            
            # Compute metrics
            mse = np.mean((true_vec - pred_vec)**2)
            mae = np.mean(np.abs(true_vec - pred_vec))
            
            k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
            
            # Create 2D field plots (input, ground truth, prediction, error)
            create_2d_field_plots(true_vec, pred_vec, input_vec, triang, 
                                 sample_idx, k1, k2, mse, mae, save_dir)
            
            # Create 2D combined plot (input + ground truth + prediction on same figure)
            create_2d_combined_plot(true_vec, pred_vec, input_vec, triang,
                                   sample_idx, k1, k2, mse, mae, save_dir)
            
            # Create 1D overlay plot
            stats = create_1d_overlay_plot(true_vec, pred_vec, input_vec, 
                                         sample_idx, k1, k2, mse, mae, save_dir)
            
            # Create detailed error analysis
            create_detailed_error_analysis_plot(true_vec, pred_vec, input_vec, 
                                               sample_idx, k1, k2, mse, mae, save_dir)
            
            all_stats.append(stats)
    
    # Create summary comparison plot
    create_summary_overlay_plot(all_stats, sample_indices, ks_val, save_dir)
    
    # Create plot index
    create_overlay_plot_index(sample_indices, ks_val, save_dir)
    
    print(f"\n✅ All comprehensive plots created!")
    print(f"📁 Directory: {save_dir}")
    print(f"📊 Plot types per sample: 2D individual (input, ground truth, prediction, error), 2D combined (all 3 on same plot), 1D overlay, error analysis")
    print(f"📊 Formats: PDF (vector), SVG (vector), EPS (publication), PNG (raster)")
    print(f"🔍 Perfect for detailed examination with infinite zoom!")


def create_summary_overlay_plot(all_stats, sample_indices, ks_val, save_dir):
    """Create a summary plot showing all samples overlaid."""
    
    plt.figure(figsize=(24, 12))  # Much wider for better detail
    
    # Load data again for summary
    model, X_val, Y_val, ks_val, edge_index, data = load_model_and_data()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    model.eval()
    with torch.no_grad():
        for i, sample_idx in enumerate(sample_indices):
            x_sample = X_val[sample_idx].unsqueeze(-1)
            y_true = Y_val[sample_idx]
            y_pred = model(x_sample, edge_index).squeeze(-1)
            
            true_vec = y_true.numpy()
            pred_vec = y_pred.numpy()
            
            k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
            mse = all_stats[i]['mse']
            
            # Plot with offset for clarity
            offset = i * 0.5
            node_indices = range(len(true_vec))
            
            plt.plot(node_indices, true_vec + offset, color=colors[i], linewidth=1.5, 
                    alpha=0.9, label=f'Sample {sample_idx} GT (k1={k1},k2={k2})')
            plt.plot(node_indices, pred_vec + offset, color=colors[i], linewidth=1.5, 
                    linestyle='--', alpha=0.9, label=f'Sample {sample_idx} Pred (MSE={mse:.2e})')
    
    plt.title(f'All Samples Overlay Comparison\n'
              f'Solid=Ground Truth, Dashed=Predictions (vertically offset for clarity)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Node Index', fontsize=16)
    plt.ylabel('Field Value (with vertical offset)', fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.4, linewidth=0.8)
    plt.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    base_name = 'all_samples_overlay_summary'
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    
    print(f"  Summary overlay plot saved: {base_name}.[pdf/svg/png]")
    plt.close()


def create_overlay_plot_index(sample_indices, ks_val, save_dir):
    """Create an index file for the overlay plots."""
    
    index_content = f"""1D Vector Overlay Plots Index
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DIRECTORY: {save_dir}

PLOT FORMATS:
- PDF: Vector format, perfect for detailed examination and zooming
- SVG: Vector format, web-compatible
- EPS: Publication-quality vector format
- PNG: High-resolution raster format (300 DPI)

SAMPLES WITH 1D OVERLAY PLOTS:
"""
    
    for sample_idx in sample_indices:
        k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
        index_content += f"""
Sample {sample_idx} (k1={k1}, k2={k2}):
  - sample_{sample_idx:03d}_1d_overlay_k1{k1}_k2{k2}.[pdf/svg/eps/png]
    Ground truth (red) and prediction (green) overlaid on same plot
    
  - sample_{sample_idx:03d}_error_analysis_k1{k1}_k2{k2}.[pdf/svg/eps/png]
    Detailed error analysis with multiple views
"""
    
    index_content += f"""
SUMMARY PLOT:
  - all_samples_overlay_summary.[pdf/svg/png]
    All samples overlaid with vertical offsets

PLOT FEATURES:
- Ground Truth: Red solid line
- GNN Prediction: Green dashed line  
- Error Band: Gray shading (±|error|)
- Boundary Nodes: Blue circles (where input ≈ 0)
- Comprehensive statistics in text boxes
- High-resolution vector formats for zooming

USAGE:
1. Open PDF files for detailed examination
2. Zoom in to see node-by-node accuracy
3. Compare red and green lines for visual assessment
4. Use error analysis plots for quantitative analysis

TOTAL FILES: {len([f for f in os.listdir(save_dir) if f.endswith('.pdf')])} PDF files
"""
    
    with open(os.path.join(save_dir, 'overlay_plots_index.txt'), 'w') as f:
        f.write(index_content)
    
    print(f"📋 Overlay plots index created: overlay_plots_index.txt")


if __name__ == "__main__":
    main()
