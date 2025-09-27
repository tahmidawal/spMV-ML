#!/usr/bin/env python3
"""
GPU Performance Profiler for SpMV vs MMA Operations
Comprehensive benchmarking of sparse matrix-vector multiplication vs
ML-based dense matrix-matrix approximations on GPU hardware
"""

import sys
import os
sys.path.append('/Users/tahmidawal/sparse2dense/0912-SpMV-CNN2D-SparseMatrices-nonorm/fem-poisson')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import json
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.profiler
from scipy.sparse import csr_matrix

# Import FEM modules
import poisson_utils.preprocess
import global_assembl_poisson.poisson


@dataclass
class PerformanceProfile:
    """Container for performance profiling results"""
    operation_name: str
    timing_stats: Dict[str, float]
    memory_stats: Dict[str, float]
    throughput_stats: Dict[str, float]
    gpu_utilization: Dict[str, float]
    metadata: Dict[str, Any]


class GPUPerformanceProfiler:
    """Comprehensive GPU performance profiler for SpMV vs MMA operations"""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize GPU performance profiler
        
        Args:
            device: Target device ('cuda', 'cpu', or 'auto')
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.sparse_matrix = None
        self.ml_models = {}
        self.profiles = {}
        
        # Performance tracking
        self.warmup_runs = 10
        self.benchmark_runs = 100
        
    def load_sparse_matrix_and_models(self, models_dir: str = None):
        """Load sparse matrix and ML models for profiling"""
        print("Loading sparse matrix and ML models...")
        
        # Load sparse matrix (same as in testing framework)
        p_order = 1
        mesh_file_path = "/Users/tahmidawal/sparse2dense/fem-poisson/unstructured-tubev2-1050.msh"
        surface_tags = [31]
        
        all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n, e_to_n_sort_idx, e_to_n_sorted, \
            bdry_indices, refel, u_exact = poisson_utils.preprocess.preprocess_poisson(
                p_order, mesh_file_path, surface_tags
            )
        
        n_pts = all_pts_gll_x.shape[0]
        
        jax_sparse_mat = global_assembl_poisson.poisson.assemble_global(
            p_order, all_pts_gll_x, all_pts_gll_y, all_pts_gll_z, e_to_n,
            refel.gauss_w_3d, refel.gll_to_gauss_1d, refel.d_gauss_1d
        )
        
        self.sparse_matrix = csr_matrix(
            (np.array(jax_sparse_mat.data), 
             np.array(jax_sparse_mat.indices), 
             np.array(jax_sparse_mat.indptr)),
            shape=(n_pts, n_pts)
        )
        
        # Convert to PyTorch sparse tensor for GPU operations
        self.torch_sparse_matrix = torch.sparse_csr_tensor(
            torch.tensor(self.sparse_matrix.indptr, dtype=torch.int64),
            torch.tensor(self.sparse_matrix.indices, dtype=torch.int64),
            torch.tensor(self.sparse_matrix.data, dtype=torch.float32),
            size=self.sparse_matrix.shape,
            device=self.device
        )
        
        print(f"Sparse matrix loaded: {self.sparse_matrix.shape}, {self.sparse_matrix.nnz:,} non-zeros")
        
        # Load ML models
        if models_dir:
            self.load_ml_models(models_dir)
    
    def load_ml_models(self, models_dir: str):
        """Load trained ML models"""
        from train_dense_spmv import DenseSpMV_TwoMatrix, DenseSpMV_SingleMatrix
        
        model_files = {
            'two_matrix_16x89': ('two_matrix_16x89_best.pth', (16, 89)),
            'two_matrix_8x178': ('two_matrix_8x178_best.pth', (8, 178)),
            'two_matrix_4x356': ('two_matrix_4x356_best.pth', (4, 356)),
            'two_matrix_2x712': ('two_matrix_2x712_best.pth', (2, 712)),
        }
        
        for model_name, (filename, matrix_shape) in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = DenseSpMV_TwoMatrix(vector_dim=1424, matrix_shape=matrix_shape)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                    model = model.to(self.device)
                    self.ml_models[model_name] = model
                    print(f"  Loaded {model_name} on {self.device}")
                except Exception as e:
                    print(f"  Failed to load {model_name}: {e}")
    
    def create_benchmark_vectors(self, batch_sizes: List[int], n_vectors_per_batch: int = 100) -> Dict[int, torch.Tensor]:
        """Create benchmark vectors for different batch sizes"""
        print("Creating benchmark vectors...")
        
        n_pts = self.sparse_matrix.shape[0]
        benchmark_vectors = {}
        
        for batch_size in batch_sizes:
            # Create random test vectors
            vectors = torch.randn(batch_size, n_pts, device=self.device, dtype=torch.float32)
            # Normalize vectors
            vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
            benchmark_vectors[batch_size] = vectors
            
        print(f"Created benchmark vectors for batch sizes: {batch_sizes}")
        return benchmark_vectors
    
    def profile_sparse_spmv(self, batch_vectors: Dict[int, torch.Tensor]) -> Dict[int, PerformanceProfile]:
        """Profile sparse matrix-vector multiplication performance"""
        print("Profiling sparse SpMV operations...")
        
        profiles = {}
        
        for batch_size, vectors in batch_vectors.items():
            print(f"  Profiling batch size {batch_size}...")
            
            # Memory stats before
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                memory_before = torch.cuda.memory_allocated()
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                for i in range(vectors.shape[0]):
                    _ = torch.sparse.mm(self.torch_sparse_matrix, vectors[i:i+1].T)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark runs
            times = []
            
            # Single vector timing
            single_vector = vectors[0]
            for _ in range(self.benchmark_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                result = torch.sparse.mm(self.torch_sparse_matrix, single_vector.unsqueeze(1))
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Batch timing
            batch_times = []
            for _ in range(min(self.benchmark_runs // batch_size, 10)):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                # Process batch (simulate real usage)
                for i in range(batch_size):
                    result = torch.sparse.mm(self.torch_sparse_matrix, vectors[i:i+1].T)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                batch_times.append((end_time - start_time) * 1000)
            
            # Memory stats after
            if self.device.type == 'cuda':
                memory_after = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - memory_before
            else:
                memory_after = memory_before = peak_memory = memory_used = 0
            
            # Compute statistics
            times = np.array(times)
            batch_times = np.array(batch_times) if batch_times else np.array([0])
            
            timing_stats = {
                'single_mean_ms': np.mean(times),
                'single_std_ms': np.std(times),
                'single_min_ms': np.min(times),
                'single_max_ms': np.max(times),
                'batch_mean_ms': np.mean(batch_times),
                'batch_std_ms': np.std(batch_times),
            }
            
            memory_stats = {
                'memory_before_mb': memory_before / 1e6,
                'memory_after_mb': memory_after / 1e6,
                'peak_memory_mb': peak_memory / 1e6,
                'memory_used_mb': memory_used / 1e6,
            }
            
            throughput_stats = {
                'single_vectors_per_sec': 1000.0 / timing_stats['single_mean_ms'],
                'batch_vectors_per_sec': batch_size * 1000.0 / timing_stats['batch_mean_ms'] if timing_stats['batch_mean_ms'] > 0 else 0,
                'gflops': self._estimate_spmv_flops() / (timing_stats['single_mean_ms'] / 1000) / 1e9,
            }
            
            profiles[batch_size] = PerformanceProfile(
                operation_name=f"sparse_spmv_batch_{batch_size}",
                timing_stats=timing_stats,
                memory_stats=memory_stats,
                throughput_stats=throughput_stats,
                gpu_utilization={},  # Would need NVIDIA tools for detailed GPU utilization
                metadata={'matrix_nnz': self.sparse_matrix.nnz, 'matrix_shape': self.sparse_matrix.shape}
            )
        
        return profiles
    
    def profile_ml_models(self, batch_vectors: Dict[int, torch.Tensor]) -> Dict[str, Dict[int, PerformanceProfile]]:
        """Profile ML model performance"""
        print("Profiling ML model operations...")
        
        all_profiles = {}
        
        for model_name, model in self.ml_models.items():
            print(f"  Profiling model: {model_name}")
            model_profiles = {}
            
            for batch_size, vectors in batch_vectors.items():
                print(f"    Batch size {batch_size}...")
                
                # Memory stats before
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated()
                
                # Warmup runs
                with torch.no_grad():
                    for _ in range(self.warmup_runs):
                        for i in range(min(vectors.shape[0], 10)):
                            _ = model(vectors[i])
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark runs
                times = []
                
                # Single vector timing
                single_vector = vectors[0]
                with torch.no_grad():
                    for _ in range(self.benchmark_runs):
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        start_time = time.perf_counter()
                        result = model(single_vector)
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)
                
                # Batch timing
                batch_times = []
                with torch.no_grad():
                    for _ in range(min(self.benchmark_runs // batch_size, 10)):
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        start_time = time.perf_counter()
                        # Process batch
                        for i in range(batch_size):
                            result = model(vectors[i])
                        
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        batch_times.append((end_time - start_time) * 1000)
                
                # Memory stats after
                if self.device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_used = peak_memory - memory_before
                else:
                    memory_after = memory_before = peak_memory = memory_used = 0
                
                # Compute statistics
                times = np.array(times)
                batch_times = np.array(batch_times) if batch_times else np.array([0])
                
                timing_stats = {
                    'single_mean_ms': np.mean(times),
                    'single_std_ms': np.std(times),
                    'single_min_ms': np.min(times),
                    'single_max_ms': np.max(times),
                    'batch_mean_ms': np.mean(batch_times),
                    'batch_std_ms': np.std(batch_times),
                }
                
                memory_stats = {
                    'memory_before_mb': memory_before / 1e6,
                    'memory_after_mb': memory_after / 1e6,
                    'peak_memory_mb': peak_memory / 1e6,
                    'memory_used_mb': memory_used / 1e6,
                }
                
                # Estimate FLOPs for dense matrix operations
                model_flops = self._estimate_model_flops(model)
                
                throughput_stats = {
                    'single_vectors_per_sec': 1000.0 / timing_stats['single_mean_ms'],
                    'batch_vectors_per_sec': batch_size * 1000.0 / timing_stats['batch_mean_ms'] if timing_stats['batch_mean_ms'] > 0 else 0,
                    'gflops': model_flops / (timing_stats['single_mean_ms'] / 1000) / 1e9,
                }
                
                model_profiles[batch_size] = PerformanceProfile(
                    operation_name=f"{model_name}_batch_{batch_size}",
                    timing_stats=timing_stats,
                    memory_stats=memory_stats,
                    throughput_stats=throughput_stats,
                    gpu_utilization={},
                    metadata={
                        'model_parameters': sum(p.numel() for p in model.parameters()),
                        'model_flops': model_flops
                    }
                )
            
            all_profiles[model_name] = model_profiles
        
        return all_profiles
    
    def _estimate_spmv_flops(self) -> float:
        """Estimate FLOPs for sparse matrix-vector multiplication"""
        # SpMV: 2 * nnz operations (multiply + add for each non-zero)
        return 2 * self.sparse_matrix.nnz
    
    def _estimate_model_flops(self, model) -> float:
        """Estimate FLOPs for ML model forward pass"""
        total_flops = 0
        
        if hasattr(model, 'W1') and hasattr(model, 'W2'):
            # TwoMatrix model: W1 @ X @ W2^T
            m, n = model.m, model.n
            # W1 @ X: m * m * n operations
            # Result @ W2^T: m * n * n operations
            total_flops = m * m * n + m * n * n
        elif hasattr(model, 'W'):
            # SingleMatrix model: W @ X
            m, n = model.m, model.n
            total_flops = m * m * n
        else:
            # Fallback estimate
            total_params = sum(p.numel() for p in model.parameters())
            total_flops = total_params  # Rough estimate
        
        return total_flops
    
    def run_comprehensive_profiling(self, batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Any]:
        """Run comprehensive performance profiling"""
        print("="*80)
        print("GPU PERFORMANCE PROFILING: SpMV vs MMA Operations")
        print("="*80)
        
        # Create benchmark vectors
        benchmark_vectors = self.create_benchmark_vectors(batch_sizes)
        
        # Profile sparse SpMV
        sparse_profiles = self.profile_sparse_spmv(benchmark_vectors)
        
        # Profile ML models
        ml_profiles = self.profile_ml_models(benchmark_vectors)
        
        # Combine all profiles
        all_profiles = {
            'sparse_spmv': sparse_profiles,
            'ml_models': ml_profiles
        }
        
        self.profiles = all_profiles
        return all_profiles
    
    def create_performance_plots(self, save_dir: str):
        """Create comprehensive performance visualization plots"""
        if not self.profiles:
            print("No profiles available for plotting")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data for plotting
        batch_sizes = sorted(self.profiles['sparse_spmv'].keys())
        
        # Create comprehensive performance plot
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig)
        
        # 1. Throughput comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        # Sparse SpMV throughput
        sparse_throughput = [self.profiles['sparse_spmv'][bs].throughput_stats['single_vectors_per_sec'] 
                           for bs in batch_sizes]
        ax1.plot(batch_sizes, sparse_throughput, 'o-', linewidth=2, markersize=8, 
                label='Sparse SpMV', color='red')
        
        # ML model throughput
        for model_name in self.profiles['ml_models']:
            ml_throughput = [self.profiles['ml_models'][model_name][bs].throughput_stats['single_vectors_per_sec']
                           for bs in batch_sizes]
            ax1.plot(batch_sizes, ml_throughput, 'o-', linewidth=2, markersize=6, 
                    label=f'ML: {model_name}', alpha=0.8)
        
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (vectors/sec)')
        ax1.set_title('Throughput Comparison: SpMV vs ML Models')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Latency comparison
        ax2 = fig.add_subplot(gs[1, 0])
        
        sparse_latency = [self.profiles['sparse_spmv'][bs].timing_stats['single_mean_ms'] 
                         for bs in batch_sizes]
        ax2.plot(batch_sizes, sparse_latency, 'o-', linewidth=2, label='Sparse SpMV', color='red')
        
        for model_name in self.profiles['ml_models']:
            ml_latency = [self.profiles['ml_models'][model_name][bs].timing_stats['single_mean_ms']
                         for bs in batch_sizes]
            ax2.plot(batch_sizes, ml_latency, 'o-', linewidth=2, label=f'ML: {model_name}', alpha=0.8)
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Latency Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Memory usage comparison
        ax3 = fig.add_subplot(gs[1, 1])
        
        if self.device.type == 'cuda':
            sparse_memory = [self.profiles['sparse_spmv'][bs].memory_stats['memory_used_mb'] 
                           for bs in batch_sizes]
            ax3.plot(batch_sizes, sparse_memory, 'o-', linewidth=2, label='Sparse SpMV', color='red')
            
            for model_name in self.profiles['ml_models']:
                ml_memory = [self.profiles['ml_models'][model_name][bs].memory_stats['memory_used_mb']
                           for bs in batch_sizes]
                ax3.plot(batch_sizes, ml_memory, 'o-', linewidth=2, label=f'ML: {model_name}', alpha=0.8)
            
            ax3.set_xlabel('Batch Size')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('GPU Memory Usage')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'GPU Memory\nProfiling\nNot Available\n(CPU Mode)', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Memory Usage (N/A for CPU)')
        
        # 4. GFLOPS comparison
        ax4 = fig.add_subplot(gs[1, 2])
        
        sparse_gflops = [self.profiles['sparse_spmv'][bs].throughput_stats['gflops'] 
                        for bs in batch_sizes]
        ax4.plot(batch_sizes, sparse_gflops, 'o-', linewidth=2, label='Sparse SpMV', color='red')
        
        for model_name in self.profiles['ml_models']:
            ml_gflops = [self.profiles['ml_models'][model_name][bs].throughput_stats['gflops']
                        for bs in batch_sizes]
            ax4.plot(batch_sizes, ml_gflops, 'o-', linewidth=2, label=f'ML: {model_name}', alpha=0.8)
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('GFLOPS')
        ax4.set_title('Computational Throughput')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Speedup analysis
        ax5 = fig.add_subplot(gs[2, :])
        
        for model_name in self.profiles['ml_models']:
            speedups = []
            for bs in batch_sizes:
                sparse_time = self.profiles['sparse_spmv'][bs].timing_stats['single_mean_ms']
                ml_time = self.profiles['ml_models'][model_name][bs].timing_stats['single_mean_ms']
                speedup = sparse_time / ml_time
                speedups.append(speedup)
            
            ax5.plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=6, 
                    label=f'ML: {model_name}', alpha=0.8)
        
        ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Sparse SpMV Baseline')
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Speedup Factor')
        ax5.set_title('ML Model Speedup vs Sparse SpMV (>1 = ML faster)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        
        # Sparse SpMV row
        sparse_best_throughput = max([self.profiles['sparse_spmv'][bs].throughput_stats['single_vectors_per_sec'] 
                                    for bs in batch_sizes])
        sparse_best_latency = min([self.profiles['sparse_spmv'][bs].timing_stats['single_mean_ms'] 
                                 for bs in batch_sizes])
        sparse_best_gflops = max([self.profiles['sparse_spmv'][bs].throughput_stats['gflops'] 
                                for bs in batch_sizes])
        
        summary_data.append([
            'Sparse SpMV',
            f"{sparse_best_throughput:.0f}",
            f"{sparse_best_latency:.3f}",
            f"{sparse_best_gflops:.1f}",
            f"{self.sparse_matrix.nnz:,}",
            "Baseline"
        ])
        
        # ML model rows
        for model_name in self.profiles['ml_models']:
            ml_best_throughput = max([self.profiles['ml_models'][model_name][bs].throughput_stats['single_vectors_per_sec']
                                    for bs in batch_sizes])
            ml_best_latency = min([self.profiles['ml_models'][model_name][bs].timing_stats['single_mean_ms']
                                 for bs in batch_sizes])
            ml_best_gflops = max([self.profiles['ml_models'][model_name][bs].throughput_stats['gflops']
                                for bs in batch_sizes])
            ml_params = self.profiles['ml_models'][model_name][batch_sizes[0]].metadata['model_parameters']
            
            best_speedup = max([self.profiles['sparse_spmv'][bs].timing_stats['single_mean_ms'] /
                               self.profiles['ml_models'][model_name][bs].timing_stats['single_mean_ms']
                               for bs in batch_sizes])
            
            summary_data.append([
                f'ML: {model_name}',
                f"{ml_best_throughput:.0f}",
                f"{ml_best_latency:.3f}",
                f"{ml_best_gflops:.1f}",
                f"{ml_params:,}",
                f"{best_speedup:.1f}x"
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Method', 'Best Throughput\n(vec/sec)', 'Best Latency\n(ms)', 
                                   'Best GFLOPS', 'Parameters/NNZ', 'Best Speedup'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        ax6.set_title('Performance Summary', pad=20, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gpu_performance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to: {save_dir}")
    
    def save_profiles(self, save_dir: str):
        """Save all performance profiles to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert profiles to serializable format
        serializable_profiles = {}
        
        for category, category_profiles in self.profiles.items():
            serializable_profiles[category] = {}
            
            if category == 'sparse_spmv':
                for batch_size, profile in category_profiles.items():
                    serializable_profiles[category][str(batch_size)] = {
                        'operation_name': profile.operation_name,
                        'timing_stats': profile.timing_stats,
                        'memory_stats': profile.memory_stats,
                        'throughput_stats': profile.throughput_stats,
                        'gpu_utilization': profile.gpu_utilization,
                        'metadata': profile.metadata
                    }
            else:  # ml_models
                for model_name, model_profiles in category_profiles.items():
                    serializable_profiles[category][model_name] = {}
                    for batch_size, profile in model_profiles.items():
                        serializable_profiles[category][model_name][str(batch_size)] = {
                            'operation_name': profile.operation_name,
                            'timing_stats': profile.timing_stats,
                            'memory_stats': profile.memory_stats,
                            'throughput_stats': profile.throughput_stats,
                            'gpu_utilization': profile.gpu_utilization,
                            'metadata': profile.metadata
                        }
        
        # Save to JSON
        profiles_file = os.path.join(save_dir, 'gpu_performance_profiles.json')
        with open(profiles_file, 'w') as f:
            json.dump(serializable_profiles, f, indent=2, default=str)
        
        print(f"Performance profiles saved to: {profiles_file}")


def main():
    """Main execution function"""
    # Create profiler
    profiler = GPUPerformanceProfiler()
    
    # Find and load most recent results directory
    results_dirs = [d for d in os.listdir('.') if d.startswith('dense_spmv_results_')]
    if results_dirs:
        latest_results = sorted(results_dirs)[-1]
        profiler.load_sparse_matrix_and_models(latest_results)
    else:
        print("No ML model results found. Loading sparse matrix only...")
        profiler.load_sparse_matrix_and_models()
    
    # Run comprehensive profiling
    batch_sizes = [1, 2, 4, 8, 16, 32] if profiler.device.type == 'cuda' else [1, 2, 4]
    profiles = profiler.run_comprehensive_profiling(batch_sizes)
    
    # Save results and create plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"gpu_performance_analysis_{timestamp}"
    
    profiler.save_profiles(save_dir)
    profiler.create_performance_plots(save_dir)
    
    print(f"\n{'='*80}")
    print("GPU PERFORMANCE PROFILING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {save_dir}")
    print(f"Device used: {profiler.device}")
    print(f"Batch sizes tested: {batch_sizes}")
    print(f"ML models profiled: {len(profiler.ml_models)}")


if __name__ == "__main__":
    main()

