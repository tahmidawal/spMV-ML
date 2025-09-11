#!/usr/bin/env python3
"""
Data utilities for loading and processing ML training data
Handles 1D-2D conversions and data loading
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


class SparseMatrixDataset(Dataset):
    """
    PyTorch Dataset for sparse matrix input-output pairs.
    Loads data from .npz or .h5 files.
    """
    
    def __init__(self, file_path, transform=None, normalize=True):
        """
        Args:
            file_path: Path to .npz or .h5 data file
            transform: Optional transform to apply to data
            normalize: Whether to normalize inputs/outputs
        """
        self.transform = transform
        self.normalize = normalize
        
        # Load data based on file extension
        if file_path.endswith('.npz'):
            data = np.load(file_path)
            self.inputs = data['inputs'].astype(np.float32)
            self.outputs = data['outputs'].astype(np.float32)
            if 'num_peaks' in data:
                self.num_peaks = data['num_peaks']
            else:
                self.num_peaks = None
        
        elif file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                self.inputs = f['inputs'][:].astype(np.float32)
                self.outputs = f['outputs'][:].astype(np.float32)
                if 'num_peaks' in f:
                    self.num_peaks = f['num_peaks'][:]
                else:
                    self.num_peaks = None
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Compute normalization statistics
        if normalize:
            self.input_mean = np.mean(self.inputs)
            self.input_std = np.std(self.inputs)
            self.output_mean = np.mean(self.outputs)
            self.output_std = np.std(self.outputs)
            
            # Normalize data
            self.inputs = (self.inputs - self.input_mean) / (self.input_std + 1e-8)
            self.outputs = (self.outputs - self.output_mean) / (self.output_std + 1e-8)
        else:
            self.input_mean = 0.0
            self.input_std = 1.0
            self.output_mean = 0.0
            self.output_std = 1.0
        
        print(f"Loaded dataset: {len(self.inputs)} samples")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Output shape: {self.outputs.shape}")
        if normalize:
            print(f"  Input stats: mean={self.input_mean:.4f}, std={self.input_std:.4f}")
            print(f"  Output stats: mean={self.output_mean:.4f}, std={self.output_std:.4f}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_signal = self.inputs[idx]
        output_signal = self.outputs[idx]
        
        if self.transform:
            input_signal = self.transform(input_signal)
            output_signal = self.transform(output_signal)
        
        return torch.from_numpy(input_signal), torch.from_numpy(output_signal)
    
    def denormalize_input(self, x):
        """Denormalize input data back to original scale."""
        return x * self.input_std + self.input_mean
    
    def denormalize_output(self, y):
        """Denormalize output data back to original scale."""
        return y * self.output_std + self.output_mean


def create_data_loaders(train_path, val_path=None, batch_size=8, 
                       num_workers=0, normalize=True, train_val_split=0.8):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_path: Path to training data file
        val_path: Path to validation data file (optional)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        normalize: Whether to normalize the data
        train_val_split: If val_path is None, split training data
    
    Returns:
        train_loader, val_loader (val_loader may be None)
    """
    
    # Load training dataset
    full_dataset = SparseMatrixDataset(train_path, normalize=normalize)
    
    if val_path is not None:
        # Load separate validation dataset
        val_dataset = SparseMatrixDataset(val_path, normalize=normalize)
        train_dataset = full_dataset
    else:
        # Split training dataset
        dataset_size = len(full_dataset)
        train_size = int(train_val_split * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Split dataset: {train_size} train, {val_size} validation")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


def reshape_signal_to_2d(signal, height=177, width=179):
    """
    Reshape 1D signal to 2D image with padding if necessary.
    
    Args:
        signal: 1D numpy array or tensor
        height: Target height
        width: Target width
    
    Returns:
        2D array of shape (height, width)
    """
    signal_length = len(signal) if hasattr(signal, '__len__') else signal.shape[-1]
    target_size = height * width
    
    # Convert to numpy if tensor
    if torch.is_tensor(signal):
        signal = signal.cpu().numpy()
    
    # Handle batched input
    if len(signal.shape) > 1:
        batch_size = signal.shape[0]
        signal_length = signal.shape[1]  # Get actual signal length from shape
        result = np.zeros((batch_size, height, width))
        for i in range(batch_size):
            # Pad if necessary
            if signal_length < target_size:
                padded = np.zeros(target_size)
                padded[:signal_length] = signal[i]
            else:
                padded = signal[i][:target_size]
            
            result[i] = padded.reshape(height, width)
        return result
    else:
        # Single signal
        if signal_length < target_size:
            padded = np.zeros(target_size)
            padded[:signal_length] = signal
        else:
            padded = signal[:target_size]
        
        return padded.reshape(height, width)


def reshape_2d_to_signal(image_2d, signal_length=31602):
    """
    Reshape 2D image back to 1D signal, removing padding if necessary.
    
    Args:
        image_2d: 2D array or tensor
        signal_length: Target signal length
    
    Returns:
        1D array of length signal_length
    """
    # Convert to numpy if tensor
    if torch.is_tensor(image_2d):
        image_2d = image_2d.cpu().numpy()
    
    # Handle batched input
    if len(image_2d.shape) > 2:
        batch_size = image_2d.shape[0]
        result = np.zeros((batch_size, signal_length))
        for i in range(batch_size):
            flat = image_2d[i].flatten()
            result[i] = flat[:signal_length]
        return result
    else:
        # Single image
        flat = image_2d.flatten()
        return flat[:signal_length]


def compute_relative_error(pred, target):
    """
    Compute relative L2 error between prediction and target.
    
    Args:
        pred: Predicted values (numpy array or tensor)
        target: Target values (numpy array or tensor)
    
    Returns:
        Relative error (scalar)
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Flatten if needed
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # Compute relative error
    error = np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)
    
    return error


def visualize_batch(inputs, outputs, predictions=None, n_samples=4):
    """
    Create visualization of batch samples in both 1D and 2D.
    
    Args:
        inputs: Input signals (batch_size, signal_length)
        outputs: Target outputs (batch_size, signal_length)
        predictions: Model predictions (optional)
        n_samples: Number of samples to visualize
    
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Convert to numpy if needed
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    if torch.is_tensor(outputs):
        outputs = outputs.cpu().numpy()
    if predictions is not None and torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    n_samples = min(n_samples, len(inputs))
    
    # Create figure
    n_rows = n_samples
    n_cols = 6 if predictions is not None else 4
    
    fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    
    for i in range(n_samples):
        # Reshape to 2D
        input_2d = reshape_signal_to_2d(inputs[i])
        output_2d = reshape_signal_to_2d(outputs[i])
        
        # 1D input
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(inputs[i][:1000], 'b-', linewidth=0.5)
        ax.set_title(f'Input {i+1} (1D)', fontsize=10)
        ax.set_xlabel('Index')
        ax.grid(True, alpha=0.3)
        
        # 2D input
        ax = fig.add_subplot(gs[i, 1])
        im = ax.imshow(input_2d, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Input {i+1} (2D)', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 1D output
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(outputs[i][:1000], 'r-', linewidth=0.5)
        ax.set_title(f'Target {i+1} (1D)', fontsize=10)
        ax.set_xlabel('Index')
        ax.grid(True, alpha=0.3)
        
        # 2D output
        ax = fig.add_subplot(gs[i, 3])
        im = ax.imshow(output_2d, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Target {i+1} (2D)', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        if predictions is not None:
            pred_2d = reshape_signal_to_2d(predictions[i])
            
            # 1D prediction
            ax = fig.add_subplot(gs[i, 4])
            ax.plot(predictions[i][:1000], 'g-', linewidth=0.5)
            ax.set_title(f'Prediction {i+1} (1D)', fontsize=10)
            ax.set_xlabel('Index')
            ax.grid(True, alpha=0.3)
            
            # 2D prediction
            ax = fig.add_subplot(gs[i, 5])
            im = ax.imshow(pred_2d, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Prediction {i+1} (2D)', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Add error info
            error = compute_relative_error(predictions[i], outputs[i])
            fig.text(0.9, 1 - (i+0.5)/n_rows, f'Error: {error:.4f}',
                    transform=fig.transFigure, fontsize=9)
    
    plt.suptitle('Batch Visualization: 1D and 2D Representations', fontsize=14, y=1.02)
    
    return fig


def test_data_utils():
    """Test data utility functions."""
    
    print("Testing data utilities...")
    
    # Test reshaping functions
    print("\n1. Testing reshape functions:")
    signal_1d = np.random.randn(31602)
    image_2d = reshape_signal_to_2d(signal_1d)
    signal_recovered = reshape_2d_to_signal(image_2d)
    
    print(f"  Original signal shape: {signal_1d.shape}")
    print(f"  2D image shape: {image_2d.shape}")
    print(f"  Recovered signal shape: {signal_recovered.shape}")
    
    # Check if conversion is lossless
    error = np.mean(np.abs(signal_1d - signal_recovered))
    print(f"  Reconstruction error: {error:.6f}")
    
    # Test batched reshaping
    print("\n2. Testing batched reshaping:")
    batch_signals = np.random.randn(4, 31602)
    batch_2d = reshape_signal_to_2d(batch_signals)
    batch_recovered = reshape_2d_to_signal(batch_2d)
    
    print(f"  Batch input shape: {batch_signals.shape}")
    print(f"  Batch 2D shape: {batch_2d.shape}")
    print(f"  Batch recovered shape: {batch_recovered.shape}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_data_utils()