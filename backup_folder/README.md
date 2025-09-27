# SpMV-ML: Sparse Matrix-Vector Multiplication Learning with Graph Neural Networks

This repository implements a Graph Neural Network (GNN) approach for learning forward sparse matrix-vector multiplication (SpMV) operations on finite element method (FEM) meshes.

## Overview

The project demonstrates how Graph Neural Networks can learn to approximate the forward operation `u = K @ f`, where:
- `K` is a sparse FEM stiffness matrix 
- `f` is a sinusoidal input field
- `u` is the output of the sparse matrix-vector multiplication

This is a simpler learning problem than solving the inverse operation and achieves excellent accuracy.

## Key Features

- **Forward SpMV Learning**: Learn `u = K @ f` instead of solving `K @ u = f`
- **Frequency-Aware Normalization**: Consistent output ranges across different input frequencies
- **Graph Neural Networks**: Leverage mesh topology for efficient learning
- **Comprehensive Visualizations**: 2D field plots and 1D vector overlays
- **High-Quality Outputs**: Vector format plots (PDF/SVG) for detailed analysis

## Core Files

### Main Scripts
- **`fem.py`** - Core FEM solver with forward SpMV dataset generation
- **`generate_fem_training_data.py`** - Dataset generation and plotting utilities
- **`train_gnn_best.py`** - Best GNN training script (Graph Attention Network)
- **`create_1d_overlay_plots.py`** - Comprehensive visualization script

### Key Directories
- **`ml_data/`** - Generated datasets and sparse matrices
- **`gnn_best_results_*/`** - Trained model weights and training history
- **`1d_overlay_plots_*/`** - Comprehensive analysis plots
- **`old_files/`** - Archived development files

## Usage

### 1. Generate Forward SpMV Dataset
```bash
python generate_fem_training_data.py --forward --num-samples 500 --nx 35 --ny 35
```

### 2. Train Graph Neural Network
```bash
python train_gnn_best.py --epochs 100 --conv-type GAT --hidden-dim 96
```

### 3. Create Detailed Visualizations
```bash
python create_1d_overlay_plots.py
```

## Results

### Model Performance
- **Architecture**: 4-layer Graph Attention Network (GAT)
- **Parameters**: 226,945 trainable parameters
- **Best Validation MSE**: 0.006907
- **Relative RMSE**: 3.53% (excellent accuracy!)
- **Improvement**: 49.3% better than simple baseline

### Key Achievements
- ✅ Successfully learned forward SpMV operation
- ✅ Achieved excellent spatial pattern matching
- ✅ Robust across different frequency combinations
- ✅ Efficient parameter usage vs fully connected approaches

## Technical Details

### Dataset Properties
- **Samples**: 500 (400 train, 100 validation)
- **Mesh**: 648 nodes, 1,258 triangular elements
- **Input**: Sinusoidal fields with frequencies k1, k2 ∈ {1,2,3,4,5}
- **Output**: Frequency-normalized SpMV results
- **Sparsity**: 98.94% sparse stiffness matrix

### Architecture Features
- **Graph structure**: Mesh connectivity defines graph edges
- **Attention mechanism**: 8-head attention for learning edge importance
- **Residual connections**: Improved gradient flow
- **Layer normalization**: Stable training
- **No edge weights**: Avoids numerical instability

## Visualizations

The repository generates comprehensive visualizations including:
- **2D Field Plots**: Input, ground truth, and prediction spatial fields
- **2D Combined Plots**: Side-by-side comparison on same figure
- **1D Vector Overlays**: Node-by-node accuracy analysis
- **Error Analysis**: Detailed accuracy metrics and error distributions
- **Training Curves**: Loss progression and convergence analysis

## Dependencies

```bash
pip install torch torch-geometric numpy scipy matplotlib
```

## Citation

If you use this code in your research, please cite:
```
@software{spMV_ML_2025,
  author = {Tahmid Awal},
  title = {SpMV-ML: Sparse Matrix-Vector Multiplication Learning with Graph Neural Networks},
  year = {2025},
  url = {https://github.com/tahmidawal/spMV-ML}
}
```

## License

This project is open source and available under the MIT License.
