#!/usr/bin/env python3
"""
Systematic Testing Framework for ML-based SpMV Replacement
Compare sparse matrix-vector multiplication vs ML matrix-matrix approximations
Focus on accuracy, performance, and robustness across different scenarios
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
import h5py
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

# Import FEM modules
import poisson_utils.preprocess
import global_assembl_poisson.poisson


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    robustness_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class SystematicTester:
    """Comprehensive testing framework for ML-based SpMV replacement"""
    
    def __init__(self, sparse_matrix_path: str = None, ml_models_dir: str = None):
        """
        Initialize testing framework
        
        Args:
            sparse_matrix_path: Path to sparse matrix data
            ml_models_dir: Directory containing trained ML models
        """
        self.sparse_matrix = None
        self.ml_models = {}
        self.test_results = []
        self.reference_data = {}
        
        if sparse_matrix_path:
            self.load_sparse_matrix(sparse_matrix_path)
        if ml_models_dir:
            self.load_ml_models(ml_models_dir)
    
    def load_sparse_matrix(self, matrix_path: str = None):
        """Load the FEM sparse matrix for testing"""
        print("Loading FEM Poisson sparse matrix for testing...")
        
        # Configuration for ORDER 1
        p_order = 1
        mesh_file_path = "/Users/tahmidawal/sparse2dense/fem-poisson/unstructured-tubev2-1050.msh"
        surface_tags = [31]
        
        # Load and assemble matrix
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
        
        self.matrix_properties = {
            'shape': self.sparse_matrix.shape,
            'nnz': self.sparse_matrix.nnz,
            'density': self.sparse_matrix.nnz / (n_pts * n_pts),
            'condition_number': None,  # Computed on demand
            'spectral_radius': None    # Computed on demand
        }
        
        print(f"Sparse matrix loaded: {self.sparse_matrix.shape}, {self.sparse_matrix.nnz:,} non-zeros")
        print(f"Density: {self.matrix_properties['density']:.2e}")
    
    def load_ml_models(self, models_dir: str):
        """Load trained ML models for comparison"""
        print(f"Loading ML models from {models_dir}...")
        
        # Import model classes
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
                    if 'two_matrix' in model_name:
                        model = DenseSpMV_TwoMatrix(vector_dim=1424, matrix_shape=matrix_shape)
                    else:
                        model = DenseSpMV_SingleMatrix(vector_dim=1424, matrix_shape=matrix_shape)
                    
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                    self.ml_models[model_name] = model
                    print(f"  Loaded {model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
                except Exception as e:
                    print(f"  Failed to load {model_name}: {e}")
        
        print(f"Successfully loaded {len(self.ml_models)} ML models")
    
    def create_test_vectors(self, n_vectors: int = 1000, test_type: str = 'comprehensive') -> np.ndarray:
        """
        Create diverse test vectors for systematic evaluation
        
        Args:
            n_vectors: Number of test vectors to generate
            test_type: Type of test vectors ('comprehensive', 'frequency', 'boundary', 'random')
        """
        print(f"Creating {n_vectors} test vectors (type: {test_type})...")
        
        n_pts = self.sparse_matrix.shape[0]
        vectors = []
        
        if test_type == 'comprehensive':
            # Mix of different vector types
            types = ['sinusoidal', 'polynomial', 'exponential', 'step', 'random', 'boundary']
            vectors_per_type = n_vectors // len(types)
            
            for vec_type in types:
                for i in range(vectors_per_type):
                    vector = self._create_single_test_vector(n_pts, vec_type, i)
                    vectors.append(vector)
            
            # Fill remaining with random vectors
            while len(vectors) < n_vectors:
                vector = self._create_single_test_vector(n_pts, 'random', len(vectors))
                vectors.append(vector)
        
        elif test_type == 'frequency':
            # Focus on different frequency components
            for i in range(n_vectors):
                freq = 1 + (i / n_vectors) * 50  # Frequency range 1-50
                vector = self._create_sinusoidal_vector(n_pts, freq)
                vectors.append(vector)
        
        elif test_type == 'boundary':
            # Focus on boundary condition effects
            for i in range(n_vectors):
                vector = self._create_boundary_test_vector(n_pts, i)
                vectors.append(vector)
        
        else:  # random
            for i in range(n_vectors):
                vector = np.random.randn(n_pts)
                vector = vector / np.linalg.norm(vector)  # Normalize
                vectors.append(vector)
        
        return np.array(vectors)
    
    def _create_single_test_vector(self, n_pts: int, vec_type: str, seed: int) -> np.ndarray:
        """Create a single test vector of specified type"""
        np.random.seed(seed)
        x = np.linspace(0, 1, n_pts)
        
        if vec_type == 'sinusoidal':
            freq = 1 + np.random.rand() * 20
            phase = np.random.rand() * 2 * np.pi
            vector = np.sin(np.pi * x) * np.sin(2 * np.pi * freq * x + phase)
        
        elif vec_type == 'polynomial':
            degree = np.random.randint(2, 6)
            coeffs = np.random.randn(degree + 1)
            vector = np.polyval(coeffs, x) * np.sin(np.pi * x)
        
        elif vec_type == 'exponential':
            rate = np.random.uniform(0.5, 5.0)
            vector = np.exp(-rate * x) * np.sin(np.pi * x)
        
        elif vec_type == 'step':
            n_steps = np.random.randint(3, 8)
            step_positions = np.sort(np.random.rand(n_steps - 1))
            step_values = np.random.randn(n_steps)
            vector = np.interp(x, np.concatenate([[0], step_positions, [1]]), 
                             np.concatenate([[step_values[0]], step_values, [step_values[-1]]]))
            vector = vector * np.sin(np.pi * x)
        
        elif vec_type == 'boundary':
            # Emphasize boundary effects
            interior_func = np.random.choice(['linear', 'quadratic', 'cubic'])
            if interior_func == 'linear':
                vector = x * (1 - x) * np.random.randn()
            elif interior_func == 'quadratic':
                vector = x * (1 - x) * (0.5 - x) * np.random.randn()
            else:  # cubic
                vector = x * (1 - x) * x * (1 - x) * np.random.randn()
        
        else:  # random
            vector = np.random.randn(n_pts)
        
        # Ensure boundary conditions (if applicable)
        if vec_type != 'random':
            vector[0] = 0
            vector[-1] = 0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        
        return vector
    
    def _create_sinusoidal_vector(self, n_pts: int, frequency: float) -> np.ndarray:
        """Create sinusoidal test vector with specific frequency"""
        x = np.linspace(0, 1, n_pts)
        vector = np.sin(np.pi * x) * np.sin(2 * np.pi * frequency * x)
        return vector / np.linalg.norm(vector)
    
    def _create_boundary_test_vector(self, n_pts: int, seed: int) -> np.ndarray:
        """Create vector emphasizing boundary condition effects"""
        np.random.seed(seed)
        x = np.linspace(0, 1, n_pts)
        
        # Different boundary-focused patterns
        patterns = [
            lambda x: x * (1 - x),                    # Quadratic
            lambda x: x**2 * (1 - x)**2,             # Quartic
            lambda x: np.sin(np.pi * x),             # Sine
            lambda x: x * (1 - x) * np.sin(np.pi * x),  # Mixed
        ]
        
        pattern = np.random.choice(patterns)
        vector = pattern(x) * np.random.randn()
        
        # Ensure zero boundary conditions
        vector[0] = 0
        vector[-1] = 0
        
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        
        return vector
    
    def test_numerical_precision(self, test_vectors: np.ndarray, 
                                precisions: List[str] = ['float32', 'float64']) -> TestResult:
        """Test numerical precision effects on SpMV vs ML models"""
        print("Testing numerical precision effects...")
        
        results = {}
        
        for precision in precisions:
            print(f"  Testing {precision} precision...")
            
            if precision == 'float32':
                dtype = np.float32
                torch_dtype = torch.float32
            else:
                dtype = np.float64
                torch_dtype = torch.float64
            
            # Convert matrix and vectors to specified precision
            sparse_matrix = self.sparse_matrix.astype(dtype)
            vectors = test_vectors.astype(dtype)
            
            # Compute ground truth SpMV
            ground_truth = []
            spmv_times = []
            
            for vector in vectors[:100]:  # Limit for speed
                start_time = time.time()
                result = sparse_matrix @ vector
                end_time = time.time()
                ground_truth.append(result)
                spmv_times.append(end_time - start_time)
            
            ground_truth = np.array(ground_truth)
            
            # Test ML models
            ml_results = {}
            for model_name, model in self.ml_models.items():
                model_results = []
                ml_times = []
                
                # Convert model to specified precision
                if precision == 'float32':
                    model = model.float()
                else:
                    model = model.double()
                
                with torch.no_grad():
                    for i, vector in enumerate(vectors[:100]):
                        torch_vector = torch.tensor(vector, dtype=torch_dtype)
                        
                        start_time = time.time()
                        prediction = model(torch_vector).numpy().astype(dtype)
                        end_time = time.time()
                        
                        model_results.append(prediction)
                        ml_times.append(end_time - start_time)
                
                model_results = np.array(model_results)
                
                # Compute accuracy metrics
                mse = np.mean((ground_truth - model_results)**2)
                mae = np.mean(np.abs(ground_truth - model_results))
                max_error = np.max(np.abs(ground_truth - model_results))
                rel_error = np.mean(np.abs(ground_truth - model_results) / 
                                  (np.abs(ground_truth) + 1e-12))
                
                ml_results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'max_error': max_error,
                    'relative_error': rel_error,
                    'mean_time': np.mean(ml_times),
                    'std_time': np.std(ml_times)
                }
            
            results[precision] = {
                'spmv_time': {'mean': np.mean(spmv_times), 'std': np.std(spmv_times)},
                'ml_models': ml_results
            }
        
        return TestResult(
            test_name="numerical_precision",
            accuracy_metrics=results,
            performance_metrics={},
            robustness_metrics={},
            metadata={'n_test_vectors': len(test_vectors), 'precisions': precisions}
        )
    
    def test_frequency_response(self, frequency_range: Tuple[float, float] = (1.0, 100.0), 
                               n_frequencies: int = 50) -> TestResult:
        """Test accuracy across different frequency ranges"""
        print(f"Testing frequency response from {frequency_range[0]} to {frequency_range[1]} Hz...")
        
        frequencies = np.logspace(np.log10(frequency_range[0]), 
                                np.log10(frequency_range[1]), n_frequencies)
        
        results = {}
        n_pts = self.sparse_matrix.shape[0]
        
        for freq in tqdm(frequencies, desc="Testing frequencies"):
            # Create sinusoidal test vector
            vector = self._create_sinusoidal_vector(n_pts, freq)
            
            # Ground truth SpMV
            ground_truth = self.sparse_matrix @ vector
            
            # Test ML models
            freq_results = {'frequency': freq}
            
            for model_name, model in self.ml_models.items():
                with torch.no_grad():
                    torch_vector = torch.tensor(vector, dtype=torch.float32)
                    prediction = model(torch_vector).numpy()
                    
                    mse = np.mean((ground_truth - prediction)**2)
                    mae = np.mean(np.abs(ground_truth - prediction))
                    rel_error = np.mean(np.abs(ground_truth - prediction) / 
                                      (np.abs(ground_truth) + 1e-12))
                    
                    freq_results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'relative_error': rel_error
                    }
            
            results[freq] = freq_results
        
        return TestResult(
            test_name="frequency_response",
            accuracy_metrics=results,
            performance_metrics={},
            robustness_metrics={},
            metadata={'frequency_range': frequency_range, 'n_frequencies': n_frequencies}
        )
    
    def test_conditioning_effects(self, n_test_cases: int = 100) -> TestResult:
        """Test sensitivity to matrix conditioning"""
        print("Testing conditioning number effects...")
        
        # This is a placeholder - would need to create matrices with different conditioning
        # For now, test with different input vector magnitudes and patterns
        
        results = {}
        n_pts = self.sparse_matrix.shape[0]
        
        # Test different vector magnitudes
        magnitudes = np.logspace(-6, 6, 13)  # 1e-6 to 1e6
        
        for mag in magnitudes:
            # Create test vector with specific magnitude
            vector = np.random.randn(n_pts)
            vector = vector / np.linalg.norm(vector) * mag
            
            # Ground truth
            ground_truth = self.sparse_matrix @ vector
            
            mag_results = {'magnitude': mag}
            
            for model_name, model in self.ml_models.items():
                with torch.no_grad():
                    torch_vector = torch.tensor(vector, dtype=torch.float32)
                    prediction = model(torch_vector).numpy()
                    
                    # Relative error is most important for conditioning
                    rel_error = np.linalg.norm(ground_truth - prediction) / \
                               (np.linalg.norm(ground_truth) + 1e-12)
                    
                    mag_results[model_name] = {'relative_error': rel_error}
            
            results[mag] = mag_results
        
        return TestResult(
            test_name="conditioning_effects",
            accuracy_metrics=results,
            performance_metrics={},
            robustness_metrics={},
            metadata={'magnitude_range': (magnitudes.min(), magnitudes.max())}
        )
    
    def run_comprehensive_test_suite(self, n_test_vectors: int = 1000) -> List[TestResult]:
        """Run the complete systematic test suite"""
        print("="*80)
        print("SYSTEMATIC ML-BASED SPMV TESTING FRAMEWORK")
        print("="*80)
        
        if self.sparse_matrix is None:
            print("Loading sparse matrix...")
            self.load_sparse_matrix()
        
        if not self.ml_models:
            print("No ML models loaded. Please load models first.")
            return []
        
        # Create test vectors
        test_vectors = self.create_test_vectors(n_test_vectors, 'comprehensive')
        
        # Run all tests
        test_results = []
        
        # 1. Numerical precision test
        test_results.append(self.test_numerical_precision(test_vectors))
        
        # 2. Frequency response test
        test_results.append(self.test_frequency_response())
        
        # 3. Conditioning effects test
        test_results.append(self.test_conditioning_effects())
        
        self.test_results = test_results
        return test_results
    
    def save_results(self, save_dir: str):
        """Save all test results to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results
        results_dict = {}
        for result in self.test_results:
            results_dict[result.test_name] = {
                'accuracy_metrics': result.accuracy_metrics,
                'performance_metrics': result.performance_metrics,
                'robustness_metrics': result.robustness_metrics,
                'metadata': result.metadata
            }
        
        results_file = os.path.join(save_dir, 'systematic_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
        # Create summary plots
        self.create_summary_plots(save_dir)
    
    def create_summary_plots(self, save_dir: str):
        """Create summary visualization plots"""
        if not self.test_results:
            return
        
        # Create comprehensive summary plot
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Plot frequency response
        freq_result = next((r for r in self.test_results if r.test_name == "frequency_response"), None)
        if freq_result:
            ax1 = fig.add_subplot(gs[0, :2])
            frequencies = list(freq_result.accuracy_metrics.keys())
            
            for model_name in self.ml_models.keys():
                mse_values = [freq_result.accuracy_metrics[f][model_name]['mse'] 
                             for f in frequencies]
                ax1.loglog(frequencies, mse_values, 'o-', label=model_name, alpha=0.7)
            
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('MSE')
            ax1.set_title('Frequency Response: MSE vs Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot precision comparison
        precision_result = next((r for r in self.test_results if r.test_name == "numerical_precision"), None)
        if precision_result:
            ax2 = fig.add_subplot(gs[0, 2:])
            
            precisions = list(precision_result.accuracy_metrics.keys())
            model_names = list(self.ml_models.keys())
            
            x = np.arange(len(model_names))
            width = 0.35
            
            for i, precision in enumerate(precisions):
                mse_values = [precision_result.accuracy_metrics[precision]['ml_models'][model]['mse']
                             for model in model_names]
                ax2.bar(x + i*width, mse_values, width, label=f'{precision}', alpha=0.7)
            
            ax2.set_xlabel('ML Models')
            ax2.set_ylabel('MSE')
            ax2.set_title('Numerical Precision Comparison')
            ax2.set_xticks(x + width/2)
            ax2.set_xticklabels(model_names, rotation=45)
            ax2.legend()
            ax2.set_yscale('log')
        
        # Plot conditioning effects
        conditioning_result = next((r for r in self.test_results if r.test_name == "conditioning_effects"), None)
        if conditioning_result:
            ax3 = fig.add_subplot(gs[1, :])
            
            magnitudes = list(conditioning_result.accuracy_metrics.keys())
            
            for model_name in self.ml_models.keys():
                rel_errors = [conditioning_result.accuracy_metrics[mag][model_name]['relative_error']
                             for mag in magnitudes]
                ax3.loglog(magnitudes, rel_errors, 'o-', label=model_name, alpha=0.7)
            
            ax3.set_xlabel('Input Vector Magnitude')
            ax3.set_ylabel('Relative Error')
            ax3.set_title('Conditioning Sensitivity: Relative Error vs Input Magnitude')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create summary table
        summary_data = []
        for model_name in self.ml_models.keys():
            row = [model_name]
            
            # Get best MSE from precision test
            if precision_result:
                best_mse = min(precision_result.accuracy_metrics['float32']['ml_models'][model_name]['mse'],
                              precision_result.accuracy_metrics['float64']['ml_models'][model_name]['mse'])
                row.append(f"{best_mse:.2e}")
            else:
                row.append("N/A")
            
            # Get frequency response range
            if freq_result:
                freq_mses = [freq_result.accuracy_metrics[f][model_name]['mse'] for f in frequencies]
                row.append(f"{min(freq_mses):.2e} - {max(freq_mses):.2e}")
            else:
                row.append("N/A")
            
            # Get conditioning sensitivity
            if conditioning_result:
                cond_errors = [conditioning_result.accuracy_metrics[mag][model_name]['relative_error']
                              for mag in magnitudes]
                row.append(f"{min(cond_errors):.2e} - {max(cond_errors):.2e}")
            else:
                row.append("N/A")
            
            summary_data.append(row)
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Model', 'Best MSE', 'Freq MSE Range', 'Conditioning Range'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'systematic_test_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plots saved to: {save_dir}")


def main():
    """Main execution function"""
    # Create tester
    tester = SystematicTester()
    
    # Load most recent results directory
    results_dirs = [d for d in os.listdir('.') if d.startswith('dense_spmv_results_')]
    if results_dirs:
        latest_results = sorted(results_dirs)[-1]
        tester.load_ml_models(latest_results)
    else:
        print("No ML model results found. Please train models first.")
        return
    
    # Run comprehensive test suite
    results = tester.run_comprehensive_test_suite(n_test_vectors=500)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"systematic_test_results_{timestamp}"
    tester.save_results(save_dir)
    
    print(f"\n{'='*80}")
    print("SYSTEMATIC TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {save_dir}")
    print(f"Number of tests completed: {len(results)}")
    print(f"ML models tested: {len(tester.ml_models)}")


if __name__ == "__main__":
    main()

