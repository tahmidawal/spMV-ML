#!/usr/bin/env python3
"""
Create high-quality vector format plots (PDF/SVG) for detailed examination.
One plot per image for maximum detail.
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
    checkpoint = torch.load('gnn_best_results_20250910_191922/best_model.pth', map_location='cpu')
    
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


def create_single_2d_field_plot(true_vec, pred_vec, input_vec, triang, sample_idx, k1, k2, 
                                mse, mae, save_dir, plot_type):
    """Create a single 2D field plot (either ground truth or prediction)."""
    
    plt.figure(figsize=(10, 8))
    
    if plot_type == 'ground_truth':
        data_vec = true_vec
        title = f'Ground Truth: K @ f'
        cmap = 'viridis'
        color = 'white'
    else:  # prediction
        data_vec = pred_vec
        title = f'GNN Prediction: Neural Network Output'
        cmap = 'viridis'
        color = 'white'
    
    # Create filled contour plot
    cf = plt.tricontourf(triang, data_vec, levels=30, cmap=cmap, extend='both')
    
    # Add contour lines for better detail
    contour_lines = plt.tricontour(triang, data_vec, levels=15, colors=color, 
                                  linewidths=0.8, alpha=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Enhanced colorbar
    cbar = plt.colorbar(cf, fraction=0.046, pad=0.04)
    cbar.set_label('Field Value', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Title and labels
    plt.title(f'{title}\n'
              f'Sample {sample_idx}: sin(2π·{k1}·x) × sin(2π·{k2}·y)\n'
              f'Range: [{data_vec.min():.4f}, {data_vec.max():.4f}]', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.gca().set_aspect('equal')
    
    # Add statistics text box
    if plot_type == 'prediction':
        stats_text = f'Accuracy Metrics:\nMSE = {mse:.2e}\nMAE = {mae:.2e}\nRMSE = {np.sqrt(mse):.2e}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_name = f'sample_{sample_idx:03d}_{plot_type}_k1{k1}_k2{k2}'
    
    # PDF (vector format)
    pdf_path = os.path.join(save_dir, f'{base_name}.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    
    # SVG (vector format)
    svg_path = os.path.join(save_dir, f'{base_name}.svg')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    
    # PNG (raster format, high resolution)
    png_path = os.path.join(save_dir, f'{base_name}.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"  {plot_type.title()} plot saved: {base_name}.[pdf/svg/png]")
    
    plt.close()


def create_single_1d_vector_plot(true_vec, pred_vec, input_vec, sample_idx, k1, k2, 
                                mse, mae, save_dir, plot_type):
    """Create a single 1D vector plot (either ground truth or prediction)."""
    
    plt.figure(figsize=(12, 6))
    
    node_indices = range(len(true_vec))
    
    if plot_type == 'ground_truth':
        data_vec = true_vec
        color = 'red'
        title = f'Ground Truth: K @ f (1D Vector)'
        label = 'Ground Truth K@f'
    else:  # prediction
        data_vec = pred_vec
        color = 'green'
        title = f'GNN Prediction: Neural Network Output (1D Vector)'
        label = 'GNN Prediction'
    
    # Main plot
    plt.plot(node_indices, data_vec, color=color, linewidth=1.5, alpha=0.8, label=label)
    
    # Highlight boundary nodes (typically first ~36 nodes)
    boundary_nodes = np.where(np.abs(input_vec) < 1e-10)[0]
    if len(boundary_nodes) > 0:
        plt.scatter(boundary_nodes, data_vec[boundary_nodes], 
                   c='blue', s=20, alpha=0.7, label=f'Boundary Nodes ({len(boundary_nodes)})')
    
    # Title and labels
    plt.title(f'{title}\n'
              f'Sample {sample_idx}: sin(2π·{k1}·x) × sin(2π·{k2}·y)\n'
              f'Range: [{data_vec.min():.4f}, {data_vec.max():.4f}]', 
              fontsize=14, fontweight='bold')
    
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Field Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    if plot_type == 'prediction':
        stats_text = f'Accuracy Metrics:\nMSE = {mse:.2e}\nMAE = {mae:.2e}\nRMSE = {np.sqrt(mse):.2e}\nNodes = {len(data_vec)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    # Add input information
    input_stats = f'Input Field:\nRange: [{input_vec.min():.3f}, {input_vec.max():.3f}]\nFreq: k1={k1}, k2={k2}'
    plt.text(0.98, 0.98, input_stats, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_name = f'sample_{sample_idx:03d}_{plot_type}_1d_k1{k1}_k2{k2}'
    
    # PDF (vector format)
    pdf_path = os.path.join(save_dir, f'{base_name}.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    
    # SVG (vector format)
    svg_path = os.path.join(save_dir, f'{base_name}.svg')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    
    # PNG (raster format, high resolution)
    png_path = os.path.join(save_dir, f'{base_name}.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"  {plot_type.title()} 1D plot saved: {base_name}.[pdf/svg/png]")
    
    plt.close()


def create_single_overlay_plot(true_vec, pred_vec, input_vec, sample_idx, k1, k2, 
                              mse, mae, save_dir):
    """Create a single overlay comparison plot."""
    
    plt.figure(figsize=(12, 8))
    
    node_indices = range(len(true_vec))
    error_vec = np.abs(true_vec - pred_vec)
    
    # Main overlay plot
    plt.plot(node_indices, true_vec, 'r-', linewidth=2, alpha=0.8, label='Ground Truth K@f')
    plt.plot(node_indices, pred_vec, 'g--', linewidth=2, alpha=0.8, label='GNN Prediction')
    
    # Error shading
    plt.fill_between(node_indices, true_vec - error_vec, true_vec + error_vec, 
                    alpha=0.2, color='gray', label='Error Band')
    
    # Highlight boundary nodes
    boundary_nodes = np.where(np.abs(input_vec) < 1e-10)[0]
    if len(boundary_nodes) > 0:
        plt.scatter(boundary_nodes, true_vec[boundary_nodes], 
                   c='blue', s=30, alpha=0.7, marker='o', label=f'Boundary Nodes ({len(boundary_nodes)})')
    
    # Title and labels
    plt.title(f'Direct Comparison: Ground Truth vs GNN Prediction\n'
              f'Sample {sample_idx}: sin(2π·{k1}·x) × sin(2π·{k2}·y)\n'
              f'Red = Ground Truth, Green = Prediction, Gray = Error Band', 
              fontsize=14, fontweight='bold')
    
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Field Value', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add comprehensive statistics
    correlation = np.corrcoef(true_vec, pred_vec)[0, 1]
    relative_rmse = np.sqrt(mse) / (true_vec.max() - true_vec.min())
    
    stats_text = (f'Accuracy Metrics:\n'
                 f'MSE = {mse:.2e}\n'
                 f'MAE = {mae:.2e}\n'
                 f'RMSE = {np.sqrt(mse):.2e}\n'
                 f'Correlation = {correlation:.4f}\n'
                 f'Relative RMSE = {relative_rmse:.3f}\n'
                 f'Max Error = {error_vec.max():.2e}')
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    # Add input field info
    input_info = (f'Input Field Info:\n'
                 f'Range: [{input_vec.min():.3f}, {input_vec.max():.3f}]\n'
                 f'Std: {input_vec.std():.3f}\n'
                 f'Frequency: k1={k1}, k2={k2}\n'
                 f'k1²+k2² = {k1*k1 + k2*k2}')
    
    plt.text(0.98, 0.98, input_info, transform=plt.gca().transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_name = f'sample_{sample_idx:03d}_overlay_k1{k1}_k2{k2}'
    
    # PDF (vector format)
    pdf_path = os.path.join(save_dir, f'{base_name}.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    
    # SVG (vector format)
    svg_path = os.path.join(save_dir, f'{base_name}.svg')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    
    # PNG (raster format, high resolution)
    png_path = os.path.join(save_dir, f'{base_name}.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"  Overlay plot saved: {base_name}.[pdf/svg/png]")
    
    plt.close()


def create_input_field_plot(input_vec, triang, sample_idx, k1, k2, save_dir):
    """Create input field visualization."""
    
    plt.figure(figsize=(10, 8))
    
    # 2D input field
    cf = plt.tricontourf(triang, input_vec, levels=25, cmap='RdBu_r', extend='both')
    
    # Add contour lines
    contour_lines = plt.tricontour(triang, input_vec, levels=12, colors='black', 
                                  linewidths=0.6, alpha=0.8)
    plt.clabel(contour_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Highlight zero boundary points
    boundary_mask = np.abs(input_vec) < 1e-10
    if np.any(boundary_mask):
        boundary_points_x = triang.x[boundary_mask]
        boundary_points_y = triang.y[boundary_mask]
        plt.scatter(boundary_points_x, boundary_points_y, 
                   c='white', s=25, edgecolors='black', linewidths=1, 
                   marker='s', alpha=0.9, label=f'Boundary Zeros ({np.sum(boundary_mask)})')
        plt.legend(fontsize=10)
    
    # Enhanced colorbar
    cbar = plt.colorbar(cf, fraction=0.046, pad=0.04)
    cbar.set_label('Input Field Value f(x,y)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.title(f'Input Field: f(x,y) = sin(2π·{k1}·x) × sin(2π·{k2}·y)\n'
              f'Sample {sample_idx} | Range: [{input_vec.min():.4f}, {input_vec.max():.4f}]', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_name = f'sample_{sample_idx:03d}_input_field_k1{k1}_k2{k2}'
    
    # PDF, SVG, PNG
    for fmt in ['pdf', 'svg', 'png']:
        save_path = os.path.join(save_dir, f'{base_name}.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
    
    print(f"  Input field plot saved: {base_name}.[pdf/svg/png]")
    plt.close()


def main():
    print("=== CREATING HIGH-QUALITY VECTOR FORMAT PLOTS ===")
    print()
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"vector_plots_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created directory: {save_dir}")
    
    # Load model and data
    model, X_val, Y_val, ks_val, edge_index, data = load_model_and_data()
    
    # Create triangulation for 2D plotting
    points = data['points']
    triangles = data['triangles']
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    
    # Select 5 diverse samples
    np.random.seed(42)
    n_samples = 5
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    print(f"Selected samples for detailed analysis: {sample_indices}")
    print()
    
    # Run inference and create individual plots
    model.eval()
    
    print("Creating individual high-quality plots...")
    
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
            
            # Create input field plot
            create_input_field_plot(input_vec, triang, sample_idx, k1, k2, save_dir)
            
            # Create 2D field plots (ground truth and prediction separately)
            create_single_2d_field_plot(true_vec, pred_vec, input_vec, triang, 
                                       sample_idx, k1, k2, mse, mae, save_dir, 'ground_truth')
            create_single_2d_field_plot(true_vec, pred_vec, input_vec, triang, 
                                       sample_idx, k1, k2, mse, mae, save_dir, 'prediction')
            
            # Create 1D vector plots (ground truth and prediction separately)
            create_single_1d_vector_plot(true_vec, pred_vec, input_vec, 
                                        sample_idx, k1, k2, mse, mae, save_dir, 'ground_truth')
            create_single_1d_vector_plot(true_vec, pred_vec, input_vec, 
                                        sample_idx, k1, k2, mse, mae, save_dir, 'prediction')
            
            # Create overlay comparison
            create_single_overlay_plot(true_vec, pred_vec, input_vec, 
                                     sample_idx, k1, k2, mse, mae, save_dir)
    
    # Create a summary index file
    create_plot_index(sample_indices, ks_val, save_dir)
    
    print(f"\n✅ All high-quality vector plots created!")
    print(f"📁 Directory: {save_dir}")
    print(f"📊 Total files: {len(os.listdir(save_dir))} (PDF, SVG, PNG formats)")
    print(f"🔍 Ready for detailed examination and zooming!")


def create_plot_index(sample_indices, ks_val, save_dir):
    """Create an index file listing all generated plots."""
    
    index_content = f"""High-Quality Vector Format Plots Index
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DIRECTORY: {save_dir}

PLOT TYPES:
- PDF: Vector format, perfect for zooming and printing
- SVG: Vector format, web-compatible
- PNG: High-resolution raster format (300 DPI)

SAMPLES ANALYZED:
"""
    
    for sample_idx in sample_indices:
        k1, k2 = int(ks_val[sample_idx][0]), int(ks_val[sample_idx][1])
        index_content += f"""
Sample {sample_idx} (k1={k1}, k2={k2}):
  - sample_{sample_idx:03d}_input_field_k1{k1}_k2{k2}.[pdf/svg/png]
  - sample_{sample_idx:03d}_ground_truth_k1{k1}_k2{k2}.[pdf/svg/png]
  - sample_{sample_idx:03d}_prediction_k1{k1}_k2{k2}.[pdf/svg/png]
  - sample_{sample_idx:03d}_ground_truth_1d_k1{k1}_k2{k2}.[pdf/svg/png]
  - sample_{sample_idx:03d}_prediction_1d_k1{k1}_k2{k2}.[pdf/svg/png]
  - sample_{sample_idx:03d}_overlay_k1{k1}_k2{k2}.[pdf/svg/png]
"""
    
    index_content += f"""
TOTAL FILES: {len(os.listdir(save_dir))}

PLOT DESCRIPTIONS:
1. input_field: Input sinusoidal field f(x,y)
2. ground_truth: True result of K @ f (2D field)
3. prediction: GNN predicted result (2D field)
4. ground_truth_1d: True result as 1D vector
5. prediction_1d: GNN prediction as 1D vector
6. overlay: Direct comparison (red=true, green=predicted)

USAGE:
- Open PDF files for detailed examination and zooming
- Use SVG files for web viewing or vector graphics editing
- PNG files for presentations and documentation
"""
    
    with open(os.path.join(save_dir, 'plots_index.txt'), 'w') as f:
        f.write(index_content)
    
    print(f"📋 Plot index created: {os.path.join(save_dir, 'plots_index.txt')}")


if __name__ == "__main__":
    main()
