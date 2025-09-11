#!/usr/bin/env python3
"""
Analyze GCN architecture to understand matrix operations.
Breaks down dense matrix-matrix vs sparse matrix-vector calculations.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
from scipy.sparse import load_npz
from torch_geometric.utils import from_scipy_sparse_matrix


def analyze_gcn_operations():
    """Analyze the matrix operations in GCN architecture."""
    
    print("=" * 80)
    print("GCN ARCHITECTURE: MATRIX OPERATIONS ANALYSIS")
    print("=" * 80)
    print()
    
    # Load the sparse matrix to understand the graph structure
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    edge_index, edge_weight = from_scipy_sparse_matrix(K)
    
    n_nodes = K.shape[0]
    n_edges = edge_index.shape[1]
    
    print(f"Graph Structure:")
    print(f"  - Number of nodes: {n_nodes}")
    print(f"  - Number of edges: {n_edges}")
    print(f"  - Sparsity: {n_edges / (n_nodes * n_nodes) * 100:.2f}%")
    print(f"  - Average degree: {n_edges / n_nodes:.2f}")
    print()
    
    # Define the GCN architecture parameters
    input_dim = 1
    hidden_dim = 128
    output_dim = 1
    num_layers = 4
    batch_size = 8  # Training batch size
    
    print(f"GCN Architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Output dimension: {output_dim}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Batch size (training): {batch_size}")
    print()
    
    print("=" * 80)
    print("LAYER-BY-LAYER OPERATION BREAKDOWN")
    print("=" * 80)
    print()
    
    # Analyze each layer's operations
    layers_info = []
    
    # Input Projection (if using residual connections)
    print("0. Input Projection Layer (for residual connections):")
    print("   Operation: Linear(1 → 128)")
    print("   Matrix ops:")
    print("     - Weight matrix: W ∈ ℝ^(128×1)")
    print("     - Input per sample: x ∈ ℝ^(648×1)")
    print("     - Computation: y = xW^T + b")
    print("     - This is MATRIX-VECTOR multiplication: (648×1) @ (1×128) = (648×128)")
    print("     - FLOPs per sample: 648 × 128 × 2 = 165,888")
    layers_info.append(("Input Proj", 648 * 128 * 2))
    print()
    
    # GCN Convolution Layers
    print("GCN Convolution Operation Breakdown:")
    print("-" * 40)
    print("GCN formula: H' = σ(D^(-1/2) Â D^(-1/2) X W)")
    print("Where:")
    print("  - Â = A + I (adjacency matrix with self-loops)")
    print("  - D = degree matrix")
    print("  - X = node features")
    print("  - W = learnable weight matrix")
    print()
    
    print("The GCN operation consists of TWO main steps:")
    print()
    
    # Layer 1: GCN Conv
    print("1. First GCN Layer: GCNConv(128 → 128)")
    print("   Step 1: Graph aggregation (SPARSE operations)")
    print("     - Sparse matrix-vector multiplication: Â @ X")
    print("     - This is SPARSE MATRIX-VECTOR multiplication")
    print("     - FLOPs: ~{} (only non-zero entries)".format(n_edges * hidden_dim * 2))
    print("   Step 2: Linear transformation (DENSE operations)")
    print("     - Dense matrix multiplication: XW")
    print("     - Weight matrix: W ∈ ℝ^(128×128)")
    print("     - This is MATRIX-MATRIX multiplication: (648×128) @ (128×128)")
    print("     - FLOPs per sample: 648 × 128 × 128 × 2 = 21,233,664")
    layers_info.append(("GCN Layer 1", n_edges * hidden_dim * 2 + 648 * 128 * 128 * 2))
    print()
    
    # Layers 2-3: Hidden GCN layers
    for i in range(2, num_layers):
        print(f"{i}. Hidden GCN Layer: GCNConv(128 → 128)")
        print("   Step 1: Graph aggregation (SPARSE)")
        print(f"     - Sparse FLOPs: ~{n_edges * hidden_dim * 2}")
        print("   Step 2: Linear transformation (DENSE)")
        print("     - Dense FLOPs: 648 × 128 × 128 × 2 = 21,233,664")
        layers_info.append((f"GCN Layer {i}", n_edges * hidden_dim * 2 + 648 * 128 * 128 * 2))
        print()
    
    # Output layer
    print(f"{num_layers}. Output GCN Layer: GCNConv(128 → 1)")
    print("   Step 1: Graph aggregation (SPARSE)")
    print(f"     - Sparse FLOPs: ~{n_edges * hidden_dim * 2}")
    print("   Step 2: Linear transformation (DENSE)")
    print("     - Dense FLOPs: 648 × 128 × 1 × 2 = 165,888")
    layers_info.append(("Output Layer", n_edges * hidden_dim * 2 + 648 * 128 * 1 * 2))
    print()
    
    print("=" * 80)
    print("OPERATION TYPE ANALYSIS")
    print("=" * 80)
    print()
    
    # Calculate total operations
    total_sparse_flops = n_edges * hidden_dim * 2 * num_layers
    total_dense_mm_flops = 648 * 128 * 128 * 2 * (num_layers - 1)  # Hidden layers
    total_dense_mv_flops = 648 * 128 * 2 + 648 * 128 * 1 * 2  # Input proj + output
    total_flops = total_sparse_flops + total_dense_mm_flops + total_dense_mv_flops
    
    print("Per-sample FLOPs breakdown:")
    print(f"  1. Sparse Matrix-Vector ops: {total_sparse_flops:,} ({total_sparse_flops/total_flops*100:.1f}%)")
    print(f"  2. Dense Matrix-Matrix ops:  {total_dense_mm_flops:,} ({total_dense_mm_flops/total_flops*100:.1f}%)")
    print(f"  3. Dense Matrix-Vector ops:  {total_dense_mv_flops:,} ({total_dense_mv_flops/total_flops*100:.1f}%)")
    print(f"  Total FLOPs per sample:      {total_flops:,}")
    print()
    
    print("Key Observations:")
    print("-" * 40)
    print("1. SPARSE operations (Graph Aggregation):")
    print(f"   - Each GCN layer performs sparse matrix-vector multiplication")
    print(f"   - Complexity: O(|E| × d) where |E|={n_edges}, d={hidden_dim}")
    print(f"   - This leverages the graph sparsity ({n_edges / (n_nodes * n_nodes) * 100:.2f}%)")
    print()
    
    print("2. DENSE operations (Feature Transformation):")
    print(f"   - Each GCN layer applies a dense weight matrix W")
    print(f"   - Hidden layers: W ∈ ℝ^({hidden_dim}×{hidden_dim})")
    print(f"   - This is MATRIX-MATRIX multiplication for all nodes")
    print(f"   - Complexity: O(N × d²) where N={n_nodes}, d={hidden_dim}")
    print()
    
    print("3. Computational Dominance:")
    if total_dense_mm_flops > total_sparse_flops:
        print(f"   - DENSE Matrix-Matrix operations dominate ({total_dense_mm_flops/total_flops*100:.1f}% of FLOPs)")
        print(f"   - Despite sparse graph structure, feature transformation is the bottleneck")
    else:
        print(f"   - SPARSE operations are significant ({total_sparse_flops/total_flops*100:.1f}% of FLOPs)")
    print()
    
    print("=" * 80)
    print("BATCH PROCESSING ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"During training with batch_size={batch_size}:")
    print()
    print("1. Each sample in batch is processed INDEPENDENTLY through the GCN")
    print("2. No batch-level matrix-matrix operations between samples")
    print("3. Operations per batch:")
    print(f"   - Total FLOPs: {total_flops * batch_size:,}")
    print(f"   - Memory for activations: ~{n_nodes * hidden_dim * 4 * batch_size / (1024*1024):.2f} MB")
    print()
    
    print("=" * 80)
    print("MEMORY ACCESS PATTERNS")
    print("=" * 80)
    print()
    
    print("1. Sparse Operations (Graph Aggregation):")
    print("   - Irregular memory access following graph edges")
    print("   - Cache efficiency depends on graph locality")
    print("   - Benefits from CSR/COO sparse formats")
    print()
    
    print("2. Dense Operations (Weight Multiplication):")
    print("   - Regular, cache-friendly memory access")
    print("   - Can leverage BLAS/cuBLAS optimizations")
    print("   - Benefits from tensor cores on modern GPUs")
    print()
    
    print("=" * 80)
    print("COMPARISON: GCN vs TRADITIONAL SPMV")
    print("=" * 80)
    print()
    
    print("Traditional SpMV: y = K @ x")
    print(f"  - FLOPs: {n_edges * 2} (one sparse matrix-vector multiplication)")
    print(f"  - Pure sparse operation")
    print()
    
    print("GCN for SpMV learning:")
    print(f"  - FLOPs per layer: ~{(n_edges * hidden_dim * 2 + 648 * 128 * 128 * 2):,}")
    print(f"  - Combines sparse graph ops with dense feature learning")
    print(f"  - Overhead ratio: {total_flops / (n_edges * 2):.0f}x more FLOPs than direct SpMV")
    print()
    
    print("Why GCN works for SpMV:")
    print("  1. Learns complex patterns beyond linear combination")
    print("  2. Multiple layers capture multi-hop dependencies")
    print("  3. Dense transformations learn feature representations")
    print("  4. Can generalize to different input patterns")
    print()
    
    # Create a visualization of operations
    print("=" * 80)
    print("VISUAL SUMMARY")
    print("=" * 80)
    print()
    
    print("GCN Layer Operation Flow:")
    print()
    print("  Input X")
    print("  [N × d_in]")
    print("      |")
    print("      v")
    print("  +--------+--------+")
    print("  |                 |")
    print("  | Sparse          | Dense")
    print("  | Aggregation     | Transform")
    print("  |                 |")
    print("  | Ã @ X           | X @ W^T")
    print("  | (sparse MV)     | (dense MM)")
    print("  |                 |")
    print("  | O(|E| × d)      | O(N × d²)")
    print("  |                 |")
    print("  +--------+--------+")
    print("           |")
    print("           v")
    print("       Output H")
    print("       [N × d_out]")
    print()
    
    return layers_info, total_flops


def create_flops_comparison():
    """Create a comparison of FLOPs for different operations."""
    
    import matplotlib.pyplot as plt
    
    # Get the analysis
    layers_info, total_flops = analyze_gcn_operations()
    
    # Load graph info
    K = load_npz('ml_data/sparse_matrix_K_20250910_161619.npz')
    n_nodes = K.shape[0]
    n_edges = K.nnz
    hidden_dim = 128
    
    # Calculate FLOPs for different components
    sparse_flops_per_layer = n_edges * hidden_dim * 2
    dense_mm_flops_per_layer = n_nodes * hidden_dim * hidden_dim * 2
    dense_mv_flops = n_nodes * hidden_dim * 2
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart of operation types
    labels = ['Dense MM\n(Feature Transform)', 'Sparse MV\n(Graph Aggregation)', 'Dense MV\n(Input/Output)']
    sizes = [
        dense_mm_flops_per_layer * 3,  # 3 hidden layers with MM
        sparse_flops_per_layer * 4,     # 4 layers total
        dense_mv_flops * 2              # Input projection + output
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('FLOPs Distribution in GCN\n(Per Sample)', fontsize=12, fontweight='bold')
    
    # Bar chart comparing layers
    layer_names = [info[0] for info in layers_info]
    layer_flops = [info[1] for info in layers_info]
    
    ax2.bar(range(len(layer_names)), layer_flops, color=['#99ff99'] + ['#ff9999']*3 + ['#ffcc99'])
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    ax2.set_ylabel('FLOPs')
    ax2.set_title('FLOPs per Layer', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('GCN Computational Analysis: Matrix Operations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('gcn_operations_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as: gcn_operations_analysis.png")
    plt.close()


if __name__ == "__main__":
    analyze_gcn_operations()
    create_flops_comparison()
