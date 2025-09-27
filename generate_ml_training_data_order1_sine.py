#!/usr/bin/env python3
"""
Generate ML training data for order 1 FEM with sinusoidal functions
Input: Sinusoidal signals with various frequencies and combinations
Output: Sparse matrix-vector products (A·u)
"""

import sys
import os
sys.path.append('/Users/tahmidawal/sparse2dense/0912-SpMV-CNN2D-SparseMatrices-nonorm/fem-poisson')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
from tqdm import tqdm
import h5py
import time

# Import FEM modules
import poisson_utils.preprocess
import global_assembl_poisson.poisson

def create_sine_signal(n, signal_type='random', seed=None):
    """Create a sinusoidal signal with sin(πx) envelope"""
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(0, 1, n)
    envelope = np.sin(np.pi * x)  # Ensures zero boundaries
    
    if signal_type == 'random':
        # Randomly choose type
        signal_type = np.random.choice(['single', 'double', 'triple', 'weighted', 'modulated'], 
                                      p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    if signal_type == 'single':
        # Single frequency
        freq = np.random.uniform(1, 30)
        signal = envelope * np.sin(2 * np.pi * freq * x)
        info = {'type': 'single', 'freq': freq}
        
    elif signal_type == 'double':
        # Two frequencies
        freq1 = np.random.uniform(1, 20)
        freq2 = np.random.uniform(freq1 + 2, 30)
        amp1 = np.random.uniform(0.4, 1.0)
        amp2 = 1 - amp1
        signal = envelope * (amp1 * np.sin(2 * np.pi * freq1 * x) + 
                            amp2 * np.sin(2 * np.pi * freq2 * x))
        info = {'type': 'double', 'freq1': freq1, 'freq2': freq2, 'amp1': amp1, 'amp2': amp2}
        
    elif signal_type == 'triple':
        # Three frequencies
        freq1 = np.random.uniform(1, 10)
        freq2 = np.random.uniform(freq1 + 2, 20)
        freq3 = np.random.uniform(freq2 + 2, 30)
        # Random amplitudes that sum to 1
        amps = np.random.dirichlet([1, 1, 1])
        signal = envelope * (amps[0] * np.sin(2 * np.pi * freq1 * x) + 
                            amps[1] * np.sin(2 * np.pi * freq2 * x) +
                            amps[2] * np.sin(2 * np.pi * freq3 * x))
        info = {'type': 'triple', 'freq1': freq1, 'freq2': freq2, 'freq3': freq3, 
                'amps': amps.tolist()}
        
    elif signal_type == 'weighted':
        # Weighted combination with emphasis on low or high frequency
        freq1 = np.random.uniform(1, 10)
        freq2 = np.random.uniform(15, 30)
        # Bias towards either low or high frequency
        if np.random.random() < 0.5:
            w1 = np.random.uniform(0.7, 0.95)  # Low frequency dominant
        else:
            w1 = np.random.uniform(0.05, 0.3)  # High frequency dominant
        w2 = 1 - w1
        signal = envelope * (w1 * np.sin(2 * np.pi * freq1 * x) + 
                            w2 * np.sin(2 * np.pi * freq2 * x))
        info = {'type': 'weighted', 'freq1': freq1, 'freq2': freq2, 'w1': w1, 'w2': w2}
        
    elif signal_type == 'modulated':
        # Amplitude modulated signal
        carrier_freq = np.random.uniform(10, 25)
        mod_freq = np.random.uniform(1, 5)
        mod_depth = np.random.uniform(0.3, 0.8)
        signal = envelope * np.sin(2 * np.pi * carrier_freq * x) * \
                (1 + mod_depth * np.sin(2 * np.pi * mod_freq * x))
        info = {'type': 'modulated', 'carrier': carrier_freq, 'mod_freq': mod_freq, 
                'mod_depth': mod_depth}
    
    else:  # chirp
        # Frequency sweep
        f0 = np.random.uniform(1, 5)
        f1 = np.random.uniform(20, 30)
        signal = envelope * np.sin(2 * np.pi * (f0 * x + (f1 - f0) * x**2 / 2))
        info = {'type': 'chirp', 'f0': f0, 'f1': f1}
    
    return signal, info

def plot_random_samples(inputs, outputs, signal_infos, n_plots=8, save_dir='ml_data'):
    """Plot random samples to verify data generation"""
    
    n_samples = inputs.shape[0]
    n_pts = inputs.shape[1]
    
    # Select random indices
    random_indices = np.random.choice(n_samples, min(n_plots, n_samples), replace=False)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 3.5 * n_plots))
    gs = gridspec.GridSpec(n_plots, 4, hspace=0.35, wspace=0.25)
    
    for plot_idx, sample_idx in enumerate(random_indices):
        input_signal = inputs[sample_idx]
        output_signal = outputs[sample_idx]
        info = signal_infos[sample_idx]
        
        # Create title based on signal type
        if info['type'] == 'single':
            title_str = f"sin(2π·{info['freq']:.1f}x)"
        elif info['type'] == 'double':
            title_str = f"f={info['freq1']:.1f}+{info['freq2']:.1f}"
        elif info['type'] == 'triple':
            title_str = f"3 freqs: {info['freq1']:.1f},{info['freq2']:.1f},{info['freq3']:.1f}"
        elif info['type'] == 'weighted':
            title_str = f"weighted: {info['w1']:.2f}·f{info['freq1']:.1f} + {info['w2']:.2f}·f{info['freq2']:.1f}"
        elif info['type'] == 'modulated':
            title_str = f"AM: fc={info['carrier']:.1f}, fm={info['mod_freq']:.1f}"
        else:  # chirp
            title_str = f"chirp: {info['f0']:.1f}→{info['f1']:.1f} Hz"
        
        # 1. Input Signal
        ax1 = fig.add_subplot(gs[plot_idx, 0:2])
        ax1.plot(input_signal, 'b-', linewidth=0.5, alpha=0.9)
        ax1.set_title(f'Sample {sample_idx}: Input ({title_str})', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Index', fontsize=9)
        ax1.set_ylabel('Value', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax1.set_ylim([-1.1, 1.1])
        
        # Add statistics
        stats_text = f'||u||₂={np.linalg.norm(input_signal):.1f}, max={np.max(np.abs(input_signal)):.3f}'
        ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=8)
        
        # 2. Output Signal
        ax2 = fig.add_subplot(gs[plot_idx, 2:4])
        ax2.plot(output_signal, 'r-', linewidth=0.5, alpha=0.9)
        ax2.set_title(f'Sample {sample_idx}: Output (A·u)', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Index', fontsize=9)
        ax2.set_ylabel('Value', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Add statistics
        stats_text = f'||Au||₂={np.linalg.norm(output_signal):.1f}, max={np.max(np.abs(output_signal)):.3f}'
        ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=8)
        
        # Verify boundaries are near zero
        boundary_text = f'u[0]={input_signal[0]:.2e}, u[-1]={input_signal[-1]:.2e}'
        ax1.text(0.5, -0.15, boundary_text, transform=ax1.transAxes, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle(f'Order 1 FEM: Sinusoidal Training Data Samples\n'
                 f'Total samples: {n_samples}, Signal length: {n_pts}',
                 fontsize=13, fontweight='bold', y=0.995)
    
    # Save plot
    plot_path = os.path.join(save_dir, 'order1_sine_training_samples.png')
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved sample plots to: {plot_path}")

def plot_statistics(inputs, outputs, signal_infos, save_dir='ml_data'):
    """Plot statistics of the generated dataset"""
    
    n_samples = inputs.shape[0]
    n_pts = inputs.shape[1]
    
    # Count signal types
    signal_types = [info['type'] for info in signal_infos]
    type_counts = {t: signal_types.count(t) for t in set(signal_types)}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Signal type distribution
    ax = axes[0, 0]
    ax.bar(type_counts.keys(), type_counts.values(), edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Signal Types', fontsize=12, fontweight='bold')
    ax.set_xlabel('Signal Type')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    for i, (k, v) in enumerate(type_counts.items()):
        ax.text(i, v, f'{v}\n({100*v/n_samples:.1f}%)', ha='center', va='bottom')
    
    # 2. Input signal norms
    ax = axes[0, 1]
    input_norms = np.linalg.norm(inputs, axis=1)
    ax.hist(input_norms, bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax.set_title('Distribution of Input Signal Norms ||u||₂', fontsize=12, fontweight='bold')
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(input_norms), color='red', linestyle='--', 
               label=f'Mean: {np.mean(input_norms):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Output signal norms
    ax = axes[0, 2]
    output_norms = np.linalg.norm(outputs, axis=1)
    ax.hist(output_norms, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax.set_title('Distribution of Output Signal Norms ||Au||₂', fontsize=12, fontweight='bold')
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(output_norms), color='blue', linestyle='--', 
               label=f'Mean: {np.mean(output_norms):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Frequency distribution for single-frequency signals
    ax = axes[1, 0]
    single_freqs = [info['freq'] for info in signal_infos if info['type'] == 'single']
    if single_freqs:
        ax.hist(single_freqs, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.set_title('Single-Frequency Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    # 5. Norm ratio distribution
    ax = axes[1, 1]
    norm_ratios = output_norms / (input_norms + 1e-10)
    ax.hist(norm_ratios, bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax.set_title('Distribution of Norm Ratios ||Au||/||u||', fontsize=12, fontweight='bold')
    ax.set_xlabel('Norm Ratio')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(norm_ratios), color='orange', linestyle='--', 
               label=f'Mean: {np.mean(norm_ratios):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Max absolute values
    ax = axes[1, 2]
    max_inputs = np.max(np.abs(inputs), axis=1)
    max_outputs = np.max(np.abs(outputs), axis=1)
    ax.scatter(max_inputs, max_outputs, alpha=0.3, s=10)
    ax.set_title('Max Absolute Values: Input vs Output', fontsize=12, fontweight='bold')
    ax.set_xlabel('max|u|')
    ax.set_ylabel('max|Au|')
    ax.grid(True, alpha=0.3)
    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    
    plt.suptitle(f'Order 1 FEM: Sinusoidal Dataset Statistics\n'
                 f'{n_samples} Samples, Signal Length: {n_pts}',
                 fontsize=14, fontweight='bold')
    
    # Save plot
    plot_path = os.path.join(save_dir, 'order1_sine_training_statistics.png')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved statistics plots to: {plot_path}")

def plot_spectral_comparison(inputs, outputs, signal_infos, n_samples=6, save_dir='ml_data'):
    """Plot spectral analysis of random samples"""
    
    n_total = inputs.shape[0]
    n_pts = inputs.shape[1]
    
    # Select random samples
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3 * n_samples))
    
    for idx, sample_idx in enumerate(indices):
        input_signal = inputs[sample_idx]
        output_signal = outputs[sample_idx]
        info = signal_infos[sample_idx]
        
        # Compute FFT
        fft_input = np.fft.fft(input_signal)
        fft_output = np.fft.fft(output_signal)
        freqs = np.fft.fftfreq(n_pts)[:n_pts//2]
        
        # Create title
        title = f"Type: {info['type']}"
        
        # Input spectrum
        ax = axes[idx, 0]
        ax.semilogy(freqs * n_pts, np.abs(fft_input[:n_pts//2]), 'b-', linewidth=1, alpha=0.9)
        ax.set_title(f'Input Spectrum - {title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)' if idx == n_samples-1 else '')
        ax.set_ylabel('Magnitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 50])
        ax.set_ylim([1e-2, max(np.abs(fft_input[:n_pts//2])) * 2])
        
        # Output spectrum
        ax = axes[idx, 1]
        ax.semilogy(freqs * n_pts, np.abs(fft_output[:n_pts//2]), 'r-', linewidth=1, alpha=0.9)
        ax.set_title(f'Output Spectrum - {title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)' if idx == n_samples-1 else '')
        ax.set_ylabel('Magnitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 50])
        ax.set_ylim([1e-2, max(np.abs(fft_output[:n_pts//2])) * 2])
    
    plt.suptitle('Order 1 FEM: Spectral Analysis of Training Samples',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = os.path.join(save_dir, 'order1_sine_spectral_samples.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved spectral analysis to: {plot_path}")

def plot_sparse_matrix(sparse_matrix, save_dir='ml_data', title_suffix=''):
    """Plot the sparse matrix structure and properties"""
    
    print(f"   Plotting sparse matrix structure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Spy plot - matrix structure
    ax = axes[0, 0]
    ax.spy(sparse_matrix, markersize=0.1, alpha=0.6)
    ax.set_title(f'Sparse Matrix Structure{title_suffix}\n'
                 f'Shape: {sparse_matrix.shape}, NNZ: {sparse_matrix.nnz:,}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    # 2. Row-wise non-zero count
    ax = axes[0, 1]
    row_nnz = np.diff(sparse_matrix.indptr)
    ax.hist(row_nnz, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.set_title('Distribution of Non-zeros per Row', fontsize=12, fontweight='bold')
    ax.set_xlabel('Non-zeros per Row')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(row_nnz), color='red', linestyle='--', 
               label=f'Mean: {np.mean(row_nnz):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Matrix values distribution (non-zero entries)
    ax = axes[1, 0]
    ax.hist(sparse_matrix.data, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_title('Distribution of Matrix Values', fontsize=12, fontweight='bold')
    ax.set_xlabel('Matrix Entry Value')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(sparse_matrix.data), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sparse_matrix.data):.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Matrix properties summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate matrix properties
    density = sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
    bandwidth = sparse_matrix.shape[0] - 1  # Assuming worst case
    
    # Try to estimate actual bandwidth
    rows, cols = sparse_matrix.nonzero()
    if len(rows) > 0:
        bandwidth = max(np.max(np.abs(rows - cols)), 1)
    
    properties_text = f"""Matrix Properties:
    
Shape: {sparse_matrix.shape[0]} × {sparse_matrix.shape[1]}
Non-zeros: {sparse_matrix.nnz:,}
Density: {density:.2e}
Storage: {sparse_matrix.format.upper()}

Value Statistics:
Min: {np.min(sparse_matrix.data):.2e}
Max: {np.max(sparse_matrix.data):.2e}
Mean: {np.mean(sparse_matrix.data):.2e}
Std: {np.std(sparse_matrix.data):.2e}

Structure:
Bandwidth (approx): {bandwidth}
Avg NNZ/row: {np.mean(row_nnz):.1f}
Max NNZ/row: {np.max(row_nnz)}
Min NNZ/row: {np.min(row_nnz)}

Memory Usage:
Data: {sparse_matrix.data.nbytes / 1024:.1f} KB
Indices: {sparse_matrix.indices.nbytes / 1024:.1f} KB  
Indptr: {sparse_matrix.indptr.nbytes / 1024:.1f} KB
Total: {(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / 1024:.1f} KB"""
    
    ax.text(0.05, 0.95, properties_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Order 1 FEM: Sparse Matrix Analysis{title_suffix}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plot_path = os.path.join(save_dir, f'order1_sine_sparse_matrix{title_suffix.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved sparse matrix plot to: {plot_path}")
    
    return plot_path

def generate_training_data(n_samples=1000, save_path='ml_training_data_order1_sine.npz'):
    """Generate training data with sinusoidal input-output pairs for order 1 FEM"""
    
    # Configuration for ORDER 1
    p_order = 1  # Order 1 polynomial
    mesh_file_path = "/Users/tahmidawal/sparse2dense/fem-poisson/unstructured-tubev2-1050.msh"
    surface_tags = [31]
    
    print("=" * 70)
    print(f"Generating {n_samples} Sinusoidal Input-Output Pairs for Order 1 FEM")
    print("=" * 70)
    
    # Load and assemble matrix
    print("\n1. Loading mesh and assembling sparse matrix (Order 1)...")
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
    
    print(f"   Polynomial order: {p_order}")
    print(f"   Matrix: {scipy_sparse.shape}, {scipy_sparse.nnz:,} non-zeros")
    print(f"   Density: {scipy_sparse.nnz/(n_pts*n_pts):.4e}")
    print(f"   Signal length: {n_pts:,}")
    
    # Convert sparse matrix to JAX format for computation
    jax_csr = jax.experimental.sparse.CSR(
        (scipy_sparse.data, scipy_sparse.indices, scipy_sparse.indptr),
        shape=scipy_sparse.shape
    )
    
    # Generate samples
    print(f"\n2. Generating {n_samples} sinusoidal samples...")
    
    input_signals = []
    output_signals = []
    signal_infos = []
    
    # Timing variables
    spmv_times = []
    total_generation_time = 0
    
    # Use progress bar for better tracking
    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Create sinusoidal signal with unique seed
        input_signal, info = create_sine_signal(n_pts, signal_type='random', seed=20000 + i)
        
        # Time the sparse matrix-vector multiplication
        input_jax = jnp.array(input_signal)
        
        # Warm up JAX (first few iterations)
        if i < 3:
            _ = jax.experimental.sparse.csr_matvec(jax_csr, input_jax)
        
        # Time the SpMV operation
        start_time = time.time()
        output_signal = jax.experimental.sparse.csr_matvec(jax_csr, input_jax)
        end_time = time.time()
        
        spmv_time = (end_time - start_time) * 1000  # Convert to milliseconds
        spmv_times.append(spmv_time)
        
        output_np = np.array(output_signal)
        
        input_signals.append(input_signal)
        output_signals.append(output_np)
        signal_infos.append(info)
    
    # Convert to numpy arrays
    input_signals = np.array(input_signals)  # Shape: (n_samples, n_pts)
    output_signals = np.array(output_signals)  # Shape: (n_samples, n_pts)
    
    # Calculate SpMV timing statistics
    spmv_times = np.array(spmv_times)
    total_spmv_time = np.sum(spmv_times)
    avg_spmv_time = np.mean(spmv_times)
    min_spmv_time = np.min(spmv_times)
    max_spmv_time = np.max(spmv_times)
    std_spmv_time = np.std(spmv_times)
    
    # Count signal types
    type_counts = {}
    for info in signal_infos:
        t = info['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    # Print statistics
    print("\n3. Dataset statistics:")
    print(f"   Input shape: {input_signals.shape}")
    print(f"   Output shape: {output_signals.shape}")
    print(f"   Input range: [{np.min(input_signals):.6f}, {np.max(input_signals):.6f}]")
    print(f"   Output range: [{np.min(output_signals):.6f}, {np.max(output_signals):.6f}]")
    print(f"   Signal type distribution:")
    for t, count in type_counts.items():
        print(f"     {t}: {count} ({100*count/n_samples:.1f}%)")
    
    # Print SpMV timing statistics
    print(f"\n4. Sparse Matrix-Vector Multiplication Timing:")
    print(f"   Total SpMV time: {total_spmv_time:.2f} ms")
    print(f"   Average per SpMV: {avg_spmv_time:.3f} ms")
    print(f"   Min SpMV time: {min_spmv_time:.3f} ms")
    print(f"   Max SpMV time: {max_spmv_time:.3f} ms")
    print(f"   Std SpMV time: {std_spmv_time:.3f} ms")
    print(f"   Throughput: {n_samples/total_spmv_time*1000:.0f} SpMV/sec")
    print(f"   Matrix size: {scipy_sparse.shape[0]}×{scipy_sparse.shape[1]}")
    print(f"   Non-zeros: {scipy_sparse.nnz:,}")
    print(f"   Density: {scipy_sparse.nnz/(scipy_sparse.shape[0]*scipy_sparse.shape[1]):.2e}")
    
    # Verify boundary conditions
    print(f"\n5. Boundary conditions check:")
    print(f"   Input boundaries: max|u[0]|={np.max(np.abs(input_signals[:, 0])):.2e}, "
          f"max|u[-1]|={np.max(np.abs(input_signals[:, -1])):.2e}")
    print(f"   Output boundaries: max|(Au)[0]|={np.max(np.abs(output_signals[:, 0])):.2e}, "
          f"max|(Au)[-1]|={np.max(np.abs(output_signals[:, -1])):.2e}")
    
    # Save data
    print(f"\n6. Saving data to {save_path}...")
    
    # Extract directory from save_path
    save_dir = os.path.dirname(save_path)
    if not save_dir:
        save_dir = '.'
    
    # Save in both .npz and .h5 formats for flexibility
    # NPZ format
    np.savez(save_path,
             inputs=input_signals,
             outputs=output_signals,
             signal_infos=signal_infos,
             matrix_shape=scipy_sparse.shape,
             matrix_nnz=scipy_sparse.nnz,
             n_samples=n_samples,
             signal_length=n_pts,
             p_order=p_order)
    
    print(f"   Saved as .npz: {save_path}")
    
    # HDF5 format (better for large datasets and partial loading)
    h5_path = save_path.replace('.npz', '.h5')
    with h5py.File(h5_path, 'w') as f:
        # Create datasets
        f.create_dataset('inputs', data=input_signals, compression='gzip', compression_opts=4)
        f.create_dataset('outputs', data=output_signals, compression='gzip', compression_opts=4)
        
        # Store signal info as JSON strings
        info_strings = [str(info) for info in signal_infos]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('signal_infos', data=info_strings, dtype=dt)
        
        # Add metadata as attributes
        f.attrs['n_samples'] = n_samples
        f.attrs['signal_length'] = n_pts
        f.attrs['matrix_shape'] = scipy_sparse.shape
        f.attrs['matrix_nnz'] = scipy_sparse.nnz
        f.attrs['p_order'] = p_order
        f.attrs['description'] = 'Order 1 FEM: Sinusoidal functions with sin(πx) envelope'
        
    print(f"   Saved as .h5: {h5_path}")
    
    # Also save a smaller validation set
    print("\n7. Creating validation set (10% of training)...")
    n_val = n_samples // 10
    val_indices = np.random.choice(n_samples, n_val, replace=False)
    
    val_path = save_path.replace('.npz', '_validation.npz')
    val_infos = [signal_infos[i] for i in val_indices]
    np.savez(val_path,
             inputs=input_signals[val_indices],
             outputs=output_signals[val_indices],
             signal_infos=val_infos,
             matrix_shape=scipy_sparse.shape,
             matrix_nnz=scipy_sparse.nnz,
             n_samples=n_val,
             signal_length=n_pts,
             p_order=p_order)
    
    print(f"   Saved validation set ({n_val} samples): {val_path}")
    
    # Generate visualization plots
    print("\n8. Generating visualization plots...")
    plot_random_samples(input_signals, output_signals, signal_infos, n_plots=8, save_dir=save_dir)
    plot_statistics(input_signals, output_signals, signal_infos, save_dir=save_dir)
    plot_spectral_comparison(input_signals, output_signals, signal_infos, n_samples=6, save_dir=save_dir)
    
    # Plot sparse matrix structure
    print("\n9. Plotting sparse matrix structure...")
    matrix_plot_path = plot_sparse_matrix(scipy_sparse, save_dir=save_dir, title_suffix=' (Order 1 FEM)')
    
    print("\n" + "=" * 70)
    print("Data generation complete!")
    print(f"Files created:")
    print(f"  - {save_path} (training data)")
    print(f"  - {h5_path} (training data in HDF5)")
    print(f"  - {val_path} (validation data)")
    print(f"  - {os.path.join(save_dir, 'order1_sine_training_samples.png')} (sample plots)")
    print(f"  - {os.path.join(save_dir, 'order1_sine_training_statistics.png')} (statistics)")
    print(f"  - {os.path.join(save_dir, 'order1_sine_spectral_samples.png')} (spectral analysis)")
    print(f"  - {matrix_plot_path} (sparse matrix analysis)")
    print("=" * 70)
    
    return input_signals, output_signals, signal_infos, scipy_sparse

def main():
    # Generate 1000 samples by default
    output_dir = "/Users/tahmidawal/sparse2dense/0912-SpMV-CNN2D-SparseMatrices-nonorm/ml_data"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "order1_sine_1000samples.npz")
    
    # Generate the data
    inputs, outputs, signal_infos, sparse_matrix = generate_training_data(n_samples=1000, save_path=save_path)
    
    print("\n" + "=" * 70)
    print("Ready for ML training with Order 1 sinusoidal data!")
    print(f"Sparse matrix shape: {sparse_matrix.shape}, NNZ: {sparse_matrix.nnz:,}")
    print("=" * 70)

def load_and_access_sparse_matrix(data_path):
    """
    Example function showing how to access the sparse matrix from saved data.
    
    Args:
        data_path: Path to the .npz file containing training data
        
    Returns:
        tuple: (inputs, outputs, sparse_matrix_info, reconstructed_matrix)
    """
    
    print(f"Loading data from: {data_path}")
    
    # Load the data
    data = np.load(data_path, allow_pickle=True)
    
    inputs = data['inputs']
    outputs = data['outputs']
    matrix_shape = data['matrix_shape']
    matrix_nnz = data['matrix_nnz']
    
    print(f"Data loaded:")
    print(f"  - Input shape: {inputs.shape}")
    print(f"  - Output shape: {outputs.shape}")
    print(f"  - Matrix shape: {matrix_shape}")
    print(f"  - Matrix NNZ: {matrix_nnz}")
    
    # Note: The actual sparse matrix data (values, indices, indptr) is not saved
    # in the training data file. To access the matrix, you need to regenerate it
    # or save it separately. Here's how to regenerate it:
    
    print("\nTo access the actual sparse matrix, you need to:")
    print("1. Call generate_training_data() and capture the 4th return value")
    print("2. Or regenerate the matrix using the FEM assembly code")
    
    return inputs, outputs, (matrix_shape, matrix_nnz)

if __name__ == "__main__":
    main()