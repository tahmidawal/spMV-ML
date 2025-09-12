#!/usr/bin/env python3
"""
Time FEM SpMV operations for forward problems.
Measure performance of sparse matrix-vector multiplication in fem.py
similar to what's done in create_dense_model_plots.py
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import os
import json
from datetime import datetime

# Import the FEM solver
from fem import FEMPoissonSolver


def time_single_spmv(K_sparse, x_vector, num_runs=100, warmup_runs=10):
    """
    Time a single sparse matrix-vector multiplication with warmup.
    
    Parameters:
    - K_sparse: Sparse matrix
    - x_vector: Input vector
    - num_runs: Number of timing runs
    - warmup_runs: Number of warmup runs (not timed)
    
    Returns:
    - mean_time: Average time per SpMV in seconds
    - std_time: Standard deviation of timing
    - min_time: Minimum time
    - max_time: Maximum time
    """
    
    # Warmup runs
    for _ in range(warmup_runs):
        _ = K_sparse.dot(x_vector)
    
    # Timing runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = K_sparse.dot(x_vector)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'times': times,
        'result_shape': result.shape,
        'result_norm': np.linalg.norm(result)
    }


def load_fem_data():
    """Load the FEM data and sparse matrix."""
    
    # Load the dataset
    data = np.load('ml_data/fem_forward_spmv_freq_norm_k1k2_1to5_500s_20250912_012326.npz')
    
    # Load the sparse matrix
    K_sparse = load_npz('ml_data/sparse_matrix_K_20250912_012326.npz')
    
    X_val = data['X_val']
    Y_val = data['Y_val']
    ks_val = data['ks_val']
    points = data['points']
    triangles = data['triangles']
    
    print(f"Loaded FEM data:")
    print(f"  Sparse matrix K: {K_sparse.shape}, {K_sparse.nnz} non-zeros")
    print(f"  Sparsity: {100*(1-K_sparse.nnz/(K_sparse.shape[0]*K_sparse.shape[1])):.2f}%")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Vector dimension: {X_val.shape[1]}")
    
    return K_sparse, X_val, Y_val, ks_val, points, triangles, data


def benchmark_fem_spmv(K_sparse, X_val, Y_val, ks_val, num_samples=50, num_runs=100):
    """
    Benchmark FEM SpMV operations on multiple samples.
    
    Parameters:
    - K_sparse: Sparse stiffness matrix
    - X_val: Input vectors
    - Y_val: Expected output vectors (for verification)
    - ks_val: Frequency pairs for each sample
    - num_samples: Number of samples to benchmark
    - num_runs: Number of timing runs per sample
    
    Returns:
    - timing_results: Dictionary with timing statistics
    """
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING FEM SpMV OPERATIONS")
    print(f"{'='*60}")
    
    # Select samples for timing
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_val), min(num_samples, len(X_val)), replace=False)
    
    timing_results = {
        'sample_times': [],
        'sample_indices': sample_indices,
        'k_pairs': [],
        'verification_errors': [],
        'matrix_info': {
            'shape': K_sparse.shape,
            'nnz': K_sparse.nnz,
            'sparsity_percent': 100*(1-K_sparse.nnz/(K_sparse.shape[0]*K_sparse.shape[1])),
            'dtype': str(K_sparse.dtype)
        }
    }
    
    print(f"Timing {len(sample_indices)} samples with {num_runs} runs each...")
    
    for i, sample_idx in enumerate(sample_indices):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sample_indices)} samples")
        
        x_input = X_val[sample_idx]
        y_expected = Y_val[sample_idx]
        k1, k2 = ks_val[sample_idx]
        
        # Time the SpMV operation
        timing_result = time_single_spmv(K_sparse, x_input, num_runs=num_runs)
        
        # Verify correctness (compute actual SpMV and compare with expected)
        y_computed_raw = K_sparse.dot(x_input)
        
        # Apply frequency-aware normalization (same as in dataset generation)
        frequency_factor = k1*k1 + k2*k2
        reference_frequency = 1*1 + 1*1  # k1=1, k2=1 reference
        y_computed = y_computed_raw * (reference_frequency / frequency_factor)
        
        # Compute verification error
        verification_error = np.linalg.norm(y_expected - y_computed)
        
        # Store results
        timing_results['sample_times'].append(timing_result)
        timing_results['k_pairs'].append((k1, k2))
        timing_results['verification_errors'].append(verification_error)
    
    # Compute aggregate statistics
    all_times = np.concatenate([result['times'] for result in timing_results['sample_times']])
    
    timing_results['aggregate_stats'] = {
        'total_spmv_operations': len(all_times),
        'mean_time_seconds': np.mean(all_times),
        'std_time_seconds': np.std(all_times),
        'min_time_seconds': np.min(all_times),
        'max_time_seconds': np.max(all_times),
        'median_time_seconds': np.median(all_times),
        'mean_time_milliseconds': np.mean(all_times) * 1000,
        'mean_time_microseconds': np.mean(all_times) * 1e6,
        'median_time_microseconds': np.median(all_times) * 1e6,
        'std_time_microseconds': np.std(all_times) * 1e6,
        'operations_per_second': 1.0 / np.mean(all_times),
        'max_verification_error': np.max(timing_results['verification_errors']),
        'mean_verification_error': np.mean(timing_results['verification_errors'])
    }
    
    return timing_results


def create_timing_plots(timing_results, save_dir):
    """Create comprehensive timing analysis plots."""
    
    print(f"\nCreating timing analysis plots...")
    
    # Extract data
    all_times = np.concatenate([result['times'] for result in timing_results['sample_times']])
    sample_means = [result['mean_time'] for result in timing_results['sample_times']]
    k_pairs = timing_results['k_pairs']
    verification_errors = timing_results['verification_errors']
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Timing distribution
    ax1 = axes[0, 0]
    ax1.hist(all_times * 1e6, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(all_times) * 1e6, color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_times)*1e6:.1f} μs')
    ax1.axvline(np.median(all_times) * 1e6, color='orange', linestyle='--', 
                label=f'Median: {np.median(all_times)*1e6:.1f} μs')
    ax1.set_xlabel('SpMV Time (microseconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'SpMV Timing Distribution\n{len(all_times)} operations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Timing vs frequency pairs
    ax2 = axes[0, 1]
    frequency_factors = [k1*k1 + k2*k2 for k1, k2 in k_pairs]
    sample_times_us = [t * 1e6 for t in sample_means]
    
    scatter = ax2.scatter(frequency_factors, sample_times_us, 
                         c=frequency_factors, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax2, label='k1² + k2²')
    ax2.set_xlabel('Frequency Factor (k1² + k2²)')
    ax2.set_ylabel('Mean SpMV Time (μs)')
    ax2.set_title('SpMV Time vs Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Verification errors
    ax3 = axes[0, 2]
    ax3.semilogy(verification_errors, 'ro-', alpha=0.7, markersize=4)
    ax3.axhline(np.mean(verification_errors), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(verification_errors):.2e}')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Verification Error (L2 norm)')
    ax3.set_title('SpMV Verification Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics comparison
    ax4 = axes[1, 0]
    metrics = ['Mean (μs)', 'Median (μs)', 'Min (μs)', 'Max (μs)', 'Std (μs)']
    values = [
        np.mean(all_times) * 1e6,
        np.median(all_times) * 1e6,
        np.min(all_times) * 1e6,
        np.max(all_times) * 1e6,
        np.std(all_times) * 1e6
    ]
    
    bars = ax4.bar(range(len(metrics)), values, 
                   color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(metrics, rotation=45, ha='right')
    ax4.set_ylabel('Time (microseconds)')
    ax4.set_title('SpMV Performance Metrics')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Matrix characteristics
    ax5 = axes[1, 1]
    matrix_info = timing_results['matrix_info']
    
    # Create matrix info display
    info_text = f"""Matrix Information:
Shape: {matrix_info['shape'][0]}×{matrix_info['shape'][1]}
Non-zeros: {matrix_info['nnz']:,}
Sparsity: {matrix_info['sparsity_percent']:.2f}%
Data type: {matrix_info['dtype']}

Performance Summary:
Operations/sec: {timing_results['aggregate_stats']['operations_per_second']:.0f}
Mean time: {timing_results['aggregate_stats']['mean_time_microseconds']:.1f} μs
Throughput: {matrix_info['nnz'] / timing_results['aggregate_stats']['mean_time_seconds'] / 1e6:.1f} MFLOP/s
"""
    
    ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Matrix & Performance Summary')
    
    # Plot 6: Timing vs sample index (check for consistency)
    ax6 = axes[1, 2]
    sample_times_us = [result['mean_time'] * 1e6 for result in timing_results['sample_times']]
    ax6.plot(sample_times_us, 'b-', alpha=0.7, linewidth=1)
    ax6.axhline(np.mean(sample_times_us), color='red', linestyle='--', 
                label=f'Overall Mean: {np.mean(sample_times_us):.1f} μs')
    ax6.fill_between(range(len(sample_times_us)), 
                     np.mean(sample_times_us) - np.std(sample_times_us),
                     np.mean(sample_times_us) + np.std(sample_times_us),
                     alpha=0.2, color='red', label=f'±1σ: {np.std(sample_times_us):.1f} μs')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Mean SpMV Time (μs)')
    ax6.set_title('Timing Consistency Across Samples')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'FEM Sparse Matrix-Vector Multiplication Timing Analysis\n'
                 f'Matrix: {matrix_info["shape"][0]}×{matrix_info["shape"][1]}, '
                 f'{matrix_info["nnz"]:,} non-zeros, '
                 f'{matrix_info["sparsity_percent"]:.1f}% sparse',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    save_path = os.path.join(save_dir, 'fem_spmv_timing_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Timing analysis plot saved: {save_path}")
    plt.close()


def compare_with_dense_models(timing_results, save_dir):
    """Compare FEM SpMV timing with dense model inference timing."""
    
    print(f"\nComparing with dense model performance...")
    
    # Load dense model results if available
    import glob
    result_dirs = glob.glob('dense_spmv_results_*')
    
    if not result_dirs:
        print("  No dense model results found for comparison")
        return
    
    latest_dir = sorted(result_dirs)[-1]
    
    try:
        with open(os.path.join(latest_dir, 'all_results.json'), 'r') as f:
            dense_results = json.load(f)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Timing comparison
        ax1 = axes[0]
        
        fem_time = timing_results['aggregate_stats']['mean_time_microseconds']
        
        # Extract dense model inference times (if available in results)
        # Note: This assumes the dense model results contain timing info
        dense_times = []
        dense_names = []
        
        for model_name, model_results in dense_results.items():
            if 'inference_time_us' in model_results:
                dense_times.append(model_results['inference_time_us'])
                dense_names.append(model_name)
        
        if dense_times:
            # Create comparison bar plot
            all_times = [fem_time] + dense_times
            all_names = ['FEM SpMV'] + [name.replace('two_matrix_', '') for name in dense_names]
            
            bars = ax1.bar(range(len(all_names)), all_times, 
                          color=['red'] + ['blue'] * len(dense_times), alpha=0.7)
            ax1.set_xticks(range(len(all_names)))
            ax1.set_xticklabels(all_names, rotation=45, ha='right')
            ax1.set_ylabel('Time (microseconds)')
            ax1.set_title('FEM SpMV vs Dense Model Inference')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, val in zip(bars, all_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No timing data available\nin dense model results', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Dense Model Timing Comparison')
        
        # Plot 2: Accuracy vs Speed tradeoff
        ax2 = axes[1]
        
        fem_accuracy = 1.0 - timing_results['aggregate_stats']['mean_verification_error']
        
        # Plot FEM point
        ax2.scatter([fem_time], [fem_accuracy], c='red', s=100, 
                   label='FEM SpMV (Exact)', marker='o', alpha=0.8)
        
        # Add dense model points if accuracy data is available
        for model_name, model_results in dense_results.items():
            if 'val_loss' in model_results and model_name in dense_names:
                # Convert validation loss to approximate accuracy
                val_loss = model_results['val_loss']
                approx_accuracy = max(0, 1.0 - val_loss)  # Simple approximation
                
                model_time = dense_times[dense_names.index(model_name.replace('two_matrix_', ''))]
                ax2.scatter([model_time], [approx_accuracy], c='blue', s=80, 
                           alpha=0.7, label=f'Dense {model_name.replace("two_matrix_", "")}')
        
        ax2.set_xlabel('Time (microseconds)')
        ax2.set_ylabel('Accuracy (1 - error)')
        ax2.set_title('Accuracy vs Speed Tradeoff')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('FEM SpMV vs Dense Model Performance Comparison', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'fem_vs_dense_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Comparison plot saved: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"  Could not load dense model results: {e}")


def main():
    """Main timing analysis function."""
    
    print("="*80)
    print("FEM SPARSE MATRIX-VECTOR MULTIPLICATION TIMING ANALYSIS")
    print("="*80)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"fem_spmv_timing_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load FEM data
    K_sparse, X_val, Y_val, ks_val, points, triangles, data = load_fem_data()
    
    # Benchmark SpMV operations
    timing_results = benchmark_fem_spmv(
        K_sparse, X_val, Y_val, ks_val, 
        num_samples=50,  # Test on 50 samples
        num_runs=100     # 100 timing runs per sample
    )
    
    # Print summary
    stats = timing_results['aggregate_stats']
    print(f"\n{'='*60}")
    print(f"FEM SpMV TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Matrix size: {timing_results['matrix_info']['shape'][0]}×{timing_results['matrix_info']['shape'][1]}")
    print(f"Non-zeros: {timing_results['matrix_info']['nnz']:,}")
    print(f"Sparsity: {timing_results['matrix_info']['sparsity_percent']:.2f}%")
    print(f"")
    print(f"Timing Results ({stats['total_spmv_operations']} operations):")
    print(f"  Mean time: {stats['mean_time_microseconds']:.1f} μs")
    print(f"  Median time: {stats['median_time_microseconds']:.1f} μs")
    print(f"  Min time: {stats['min_time_seconds']*1e6:.1f} μs")
    print(f"  Max time: {stats['max_time_seconds']*1e6:.1f} μs")
    print(f"  Std deviation: {stats['std_time_seconds']*1e6:.1f} μs")
    print(f"")
    print(f"Performance Metrics:")
    print(f"  Operations per second: {stats['operations_per_second']:.0f}")
    print(f"  Throughput: {timing_results['matrix_info']['nnz'] / stats['mean_time_seconds'] / 1e6:.1f} MFLOP/s")
    print(f"")
    print(f"Verification:")
    print(f"  Max error: {stats['max_verification_error']:.2e}")
    print(f"  Mean error: {stats['mean_verification_error']:.2e}")
    
    # Create timing plots
    create_timing_plots(timing_results, save_dir)
    
    # Compare with dense models
    compare_with_dense_models(timing_results, save_dir)
    
    # Save detailed results
    results_file = os.path.join(save_dir, 'timing_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_json_serializable(item) for item in obj)
        else:
            return obj
    
    json_results = convert_to_json_serializable(timing_results)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Timing analysis complete!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"📊 Detailed results: {results_file}")
    
    # Performance context
    print(f"\n{'='*60}")
    print(f"PERFORMANCE CONTEXT")
    print(f"{'='*60}")
    print(f"For a {timing_results['matrix_info']['shape'][0]}-node FEM problem:")
    print(f"  • Each SpMV takes ~{stats['mean_time_microseconds']:.1f} μs")
    print(f"  • Could perform ~{stats['operations_per_second']:.0f} SpMV operations per second")
    print(f"  • Matrix has {timing_results['matrix_info']['sparsity_percent']:.1f}% sparsity")
    print(f"  • Verification error is {stats['mean_verification_error']:.2e} (excellent accuracy)")
    print(f"")
    print(f"This timing represents the 'ground truth' performance that")
    print(f"dense neural networks are trying to approximate!")


if __name__ == "__main__":
    main()
