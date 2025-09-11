#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from fem import (generate_fem_sinusoid_dataset, generate_fem_forward_dataset, 
                 plot_fem_dataset_samples, plot_forward_spmv_verification, 
                 plot_forward_spmv_samples_2d, plot_sparse_matrix_analysis, FEMPoissonSolver)


def parse_args():
    p = argparse.ArgumentParser(description="Generate or plot FEM dataset: inverse (RHS→solution) or forward (f→K@f) with k1,k2 ∈ {1..5}")
    
    # Problem type selection
    p.add_argument('--forward', action='store_true', help='Generate forward SpMV dataset (f → K@f) instead of inverse (f → u)')
    
    # Dataset generation parameters
    p.add_argument('--num-samples', type=int, default=300, help='Total number of samples to generate (default: 300)')
    p.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    p.add_argument('--nx', type=int, default=35, help='Mesh nx (default: 35)')
    p.add_argument('--ny', type=int, default=35, help='Mesh ny (default: 35)')
    p.add_argument('--mesh-type', type=str, default='unstructured', choices=['structured', 'unstructured', 'adaptive'], help='Mesh type')
    p.add_argument('--theta-min', type=float, default=0.5, help='Theta min (default: 0.5)')
    p.add_argument('--theta-max', type=float, default=3.0, help='Theta max (default: 3.0)')
    p.add_argument('--theta-regions', type=int, default=4, help='Number of smooth theta regions (default: 4)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed (default: 42)')
    p.add_argument('--out-dir', type=str, default='ml_data', help='Output directory (default: ml_data)')
    
    # Plot-only mode
    p.add_argument('--npz', type=str, default=None, help='If provided, load this dataset and plot comparisons')
    p.add_argument('--plot-num-samples', type=int, default=5, help='Number of samples per split to plot (default: 5)')
    p.add_argument('--plot-splits', type=str, default='train,val', help='Comma-separated splits to plot: train,val')
    
    return p.parse_args()
def plot_input_output_before_after(npz_path: str, num_samples: int = 5, splits=('train', 'val')):
    data = np.load(npz_path, allow_pickle=True)

    # Load arrays
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    ks_train = data['ks_train']
    ks_val = data['ks_val']
    points = data['points']
    triangles = data['triangles']
    interior_nodes = data['interior_nodes']
    theta_field = data['theta_field']
    nx = int(data['nx'])
    ny = int(data['ny'])
    mesh_type = str(data['mesh_type'])

    # Rebuild solver and assemble K once (theta fixed)
    solver = FEMPoissonSolver(nx=nx, ny=ny, mesh_type=mesh_type)
    solver.points = points
    solver.triangles = triangles
    solver.n_nodes = len(points)
    solver.n_elements = len(triangles)
    solver.identify_boundary_nodes()
    solver.assemble_system(np.zeros(solver.n_nodes), theta_field)
    K = solver.K  # CSR

    def plot_split(X, Y, ks, split_name, save_path):
        n = X.shape[0]
        idx = np.linspace(0, n - 1, min(num_samples, n), dtype=int)
        fig, axes = plt.subplots(len(idx), 2, figsize=(12, 2.6 * len(idx)))
        if len(idx) == 1:
            axes = np.array([axes])
        for row, i in enumerate(idx):
            # Slice to interior only to remove boundary zeros
            f_vec = X[i][interior_nodes]
            u_vec = Y[i][interior_nodes]
            f_after = (K @ Y[i])[interior_nodes]
            k1, k2 = ks[i]

            # Panel 1: Input vs Solution (before SpMV)
            ax1 = axes[row, 0]
            ax1.plot(f_vec, label='f (input)', linewidth=0.8)
            ax1.plot(u_vec, label='u (solution)', linewidth=0.8)
            ax1.set_title(f"{split_name} sample {i} — before SpMV (interior, k1={int(k1)}, k2={int(k2)})")
            ax1.set_xlabel('node index')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # Panel 2: Input vs K @ u (after SpMV)
            ax2 = axes[row, 1]
            ax2.plot(f_vec, label='f (input)', linewidth=0.8)
            ax2.plot(f_after, label='K @ u (reconstructed)', linewidth=0.8)
            rmse = np.sqrt(np.mean((f_after - f_vec) ** 2))
            ax2.set_title(f"{split_name} sample {i} — after SpMV (interior, RMSE={rmse:.2e})")
            ax2.set_xlabel('node index')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()  # Close the figure to free memory

    out_dir = os.path.dirname(npz_path)
    if 'train' in splits:
        plot_split(X_train, Y_train, ks_train, 'train', os.path.join(out_dir, 'train_before_after_spmv.png'))
    if 'val' in splits:
        plot_split(X_val, Y_val, ks_val, 'val', os.path.join(out_dir, 'val_before_after_spmv.png'))


def plot_forward_spmv_dataset_verification(npz_path: str, num_samples: int = 5, splits=('train', 'val')):
    """
    Plot verification for forward SpMV dataset: compare stored K@f with recomputed K@f.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Check if this is a forward SpMV dataset
    if ('problem_type' not in data or 
        (str(data['problem_type']) != 'forward_spmv' and 
         str(data['problem_type']) != 'forward_spmv_freq_normalized')):
        print("Warning: This doesn't appear to be a forward SpMV dataset.")
        return
    
    # Detect if dataset uses frequency normalization
    is_freq_normalized = str(data['problem_type']) == 'forward_spmv_freq_normalized'
    
    # Load arrays
    X_train = data['X_train']
    Y_train = data['Y_train'] 
    X_val = data['X_val']
    Y_val = data['Y_val']
    ks_train = data['ks_train']
    ks_val = data['ks_val']
    points = data['points']
    triangles = data['triangles']
    theta_field = data['theta_field']
    nx = int(data['nx'])
    ny = int(data['ny'])
    mesh_type = str(data['mesh_type'])
    
    # Rebuild solver and K matrix
    solver = FEMPoissonSolver(nx=nx, ny=ny, mesh_type=mesh_type)
    solver.points = points
    solver.triangles = triangles
    solver.n_nodes = len(points)
    solver.n_elements = len(triangles)
    solver.identify_boundary_nodes()
    
    # Reassemble K matrix (same as during dataset generation)
    dummy_f = np.zeros(solver.n_nodes)
    solver.assemble_system(dummy_f, theta_field)
    K = solver.K
    
    out_dir = os.path.dirname(npz_path)
    
    # Plot verification and 2D samples
    if 'train' in splits:
        print("Plotting training set verification...")
        plot_forward_spmv_verification(solver, X_train, Y_train, ks_train, K, 
                                       'train', num_samples, 
                                       os.path.join(out_dir, 'train_forward_verification.png'),
                                       frequency_normalized=is_freq_normalized)
        plot_forward_spmv_samples_2d(solver, X_train, Y_train, ks_train, 
                                     'train', min(3, num_samples),
                                     os.path.join(out_dir, 'train_forward_2d_samples.png'), K)
        
    if 'val' in splits:
        print("Plotting validation set verification...")
        plot_forward_spmv_verification(solver, X_val, Y_val, ks_val, K, 
                                       'val', num_samples,
                                       os.path.join(out_dir, 'val_forward_verification.png'),
                                       frequency_normalized=is_freq_normalized)
        plot_forward_spmv_samples_2d(solver, X_val, Y_val, ks_val, 
                                     'val', min(3, num_samples),
                                     os.path.join(out_dir, 'val_forward_2d_samples.png'), K)
    
    # Generate sparse matrix analysis
    print("Generating sparse matrix analysis...")
    plot_sparse_matrix_analysis(K, solver, os.path.join(out_dir, 'sparse_matrix_analysis.png'))


def main():
    args = parse_args()

    # Plot-only mode
    if args.npz:
        splits = tuple([s.strip() for s in args.plot_splits.split(',') if s.strip()])
        
        # Detect dataset type and plot accordingly
        data = np.load(args.npz, allow_pickle=True)
        is_forward = ('problem_type' in data and 
                     (str(data['problem_type']) == 'forward_spmv' or 
                      str(data['problem_type']) == 'forward_spmv_freq_normalized'))
        
        if is_forward:
            print(f"Plotting forward SpMV verification for: {args.npz} (splits: {splits}, samples per split: {args.plot_num_samples})")
            plot_forward_spmv_dataset_verification(args.npz, num_samples=args.plot_num_samples, splits=splits)
        else:
            print(f"Plotting inverse problem (before/after SpMV) for: {args.npz} (splits: {splits}, samples per split: {args.plot_num_samples})")
            plot_input_output_before_after(args.npz, num_samples=args.plot_num_samples, splits=splits)
        return

    # Dataset generation mode
    problem_type = "forward SpMV" if args.forward else "inverse (solve)"
    print(f"Generating FEM {problem_type} dataset...")
    print(f"  num_samples: {args.num_samples}")
    print(f"  k_values: 1..5")
    print(f"  train_ratio: {args.train_ratio}")
    print(f"  mesh: {args.mesh_type} (nx={args.nx}, ny={args.ny})")
    print(f"  theta: [{args.theta_min}, {args.theta_max}] with {args.theta_regions} regions (no normalization)")
    print(f"  out_dir: {args.out_dir}")

    k_values = (1, 2, 3, 4, 5)

    # Choose dataset generation function based on mode
    if args.forward:
        print("Using forward SpMV mode: learning f → K @ f")
        X_tr, Y_tr, X_val, Y_val, meta, solver = generate_fem_forward_dataset(
            num_samples=args.num_samples,
            k_values=k_values,
            train_ratio=args.train_ratio,
            nx=args.nx,
            ny=args.ny,
            mesh_type=args.mesh_type,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            theta_regions=args.theta_regions,
            seed=args.seed,
            out_dir=args.out_dir,
            dataset_name=None
        )
        
        # Generate verification plots for forward SpMV
        os.makedirs(args.out_dir, exist_ok=True)
        
        print("Plotting forward SpMV verification...")
        K = meta['K_sparse_matrix']
        
        # Adjust number of samples based on available data
        train_plot_samples = min(5, len(X_tr))
        val_plot_samples = min(5, len(X_val))
        
        plot_forward_spmv_verification(solver, X_tr, Y_tr, meta['ks_train'], K, 
                                       'train', train_plot_samples, 
                                       os.path.join(args.out_dir, 'train_forward_verification.png'),
                                       frequency_normalized=True)  # Forward mode always uses freq normalization now
        plot_forward_spmv_verification(solver, X_val, Y_val, meta['ks_val'], K, 
                                       'val', val_plot_samples,
                                       os.path.join(args.out_dir, 'val_forward_verification.png'),
                                       frequency_normalized=True)  # Forward mode always uses freq normalization now
        
        print("Plotting forward SpMV 2D samples...")
        train_2d_samples = min(3, len(X_tr))
        val_2d_samples = min(3, len(X_val))
        
        plot_forward_spmv_samples_2d(solver, X_tr, Y_tr, meta['ks_train'], 
                                     'train', train_2d_samples,
                                     os.path.join(args.out_dir, 'train_forward_2d_samples.png'), K)
        plot_forward_spmv_samples_2d(solver, X_val, Y_val, meta['ks_val'], 
                                     'val', val_2d_samples,
                                     os.path.join(args.out_dir, 'val_forward_2d_samples.png'), K)
        
        print("Plotting sparse matrix analysis...")
        plot_sparse_matrix_analysis(K, solver, os.path.join(args.out_dir, 'sparse_matrix_analysis.png'))
        
    else:
        print("Using inverse mode: learning f → u (where K @ u = f)")
        X_tr, Y_tr, X_val, Y_val, meta, solver = generate_fem_sinusoid_dataset(
            num_samples=args.num_samples,
            k_values=k_values,
            train_ratio=args.train_ratio,
            nx=args.nx,
            ny=args.ny,
            mesh_type=args.mesh_type,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            theta_regions=args.theta_regions,
            seed=args.seed,
            out_dir=args.out_dir,
            dataset_name=None
        )

        os.makedirs(args.out_dir, exist_ok=True)
        train_plot_path = os.path.join(args.out_dir, 'train_samples.png')
        val_plot_path = os.path.join(args.out_dir, 'val_samples.png')

        print("Plotting 5 training samples...")
        plot_fem_dataset_samples(
            solver,
            X_split=X_tr,
            Y_split=Y_tr,
            ks_split=meta['ks_train'],
            split_name='train',
            num_samples=5,
            save_path=train_plot_path
        )

        print("Plotting 5 validation samples...")
        plot_fem_dataset_samples(
            solver,
            X_split=X_val,
            Y_split=Y_val,
            ks_split=meta['ks_val'],
            split_name='val',
            num_samples=5,
            save_path=val_plot_path
        )

    print("Done.")


if __name__ == '__main__':
    main()


