#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_fem1d_npz import count_parameters
from model_fem1d import PositionConditionedMixer1DNet


def timestamp_dir(base: str = 'output') -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    return out_dir


def load_npz_interiors(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    X_tr = z['X_train']
    Y_tr = z['Y_train']
    X_val = z['X_val']
    Y_val = z['Y_val']
    interior_nodes = z['interior_nodes']
    # Slice interior nodes
    X_tr = X_tr[:, interior_nodes]
    Y_tr = Y_tr[:, interior_nodes]
    X_val = X_val[:, interior_nodes]
    Y_val = Y_val[:, interior_nodes]
    return X_tr.astype(np.float32), Y_tr.astype(np.float32), X_val.astype(np.float32), Y_val.astype(np.float32), interior_nodes


def standardize_train_val(X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray):
    # Compute training statistics
    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std = X_tr.std(axis=0, keepdims=True) + 1e-8
    y_mean = Y_tr.mean(axis=0, keepdims=True)
    y_std = Y_tr.std(axis=0, keepdims=True) + 1e-8
    # Apply
    X_tr_n = (X_tr - x_mean) / x_std
    Y_tr_n = (Y_tr - y_mean) / y_std
    X_val_n = (X_val - x_mean) / x_std
    Y_val_n = (Y_val - y_mean) / y_std
    stats = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
    return X_tr_n, Y_tr_n, X_val_n, Y_val_n, stats


def plot_val_examples(X_val, Y_val, Y_pred, save_path: str, k: int = 5):
    import matplotlib.pyplot as plt
    n = min(k, X_val.shape[0])
    idx = np.linspace(0, X_val.shape[0] - 1, n, dtype=int)
    plt.figure(figsize=(12, 2.4 * n))
    for i, j in enumerate(idx):
        ax = plt.subplot(n, 1, i + 1)
        mse = float(np.mean((Y_pred[j] - Y_val[j]) ** 2))
        ax.plot(Y_val[j], label='y_true', linewidth=0.9)
        ax.plot(Y_pred[j], label='y_pred', linewidth=0.9)
        ax.set_title(f'sample {j} — MSE={mse:.3e}')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_losses(train_losses, val_losses, save_path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (standardized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def main():
    # Update path to your NPZ
    data_path = 'ml_data/fem_sinusoid_k1k2_1to5_300s_20250907_203020.npz'
    out_dir = timestamp_dir('output')

    print('=' * 60)
    print('Training LocalSpMV1DNet on FEM NPZ (interior nodes, standardized)')
    print('=' * 60)
    print('data_path:', data_path)

    # Load and standardize
    X_tr, Y_tr, X_val, Y_val, interior_nodes = load_npz_interiors(data_path)
    X_tr_n, Y_tr_n, X_val_n, Y_val_n, stats = standardize_train_val(X_tr, Y_tr, X_val, Y_val)

    device = torch.device('cpu')
    model = PositionConditionedMixer1DNet(
        kernel_sizes=(1, 3, 5, 7, 11, 15),
        dilations=(1, 2, 4, 8),
        branch_channels=8,
        activation='identity',
        bias=False,
        refiner_layers=1,
        refiner_kernel_size=3,
    ).to(device)
    print('model: PositionConditionedMixer1DNet  parameters:', count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_tr_t = torch.from_numpy(X_tr_n).to(device)
    Y_tr_t = torch.from_numpy(Y_tr_n).to(device)
    X_val_t = torch.from_numpy(X_val_n).to(device)
    Y_val_t = torch.from_numpy(Y_val_n).to(device)

    n_epochs = 200
    batch_size = 64
    n_train = X_tr_t.shape[0]
    indices = np.arange(n_train)

    best_val = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, n_train, batch_size):
            idx = indices[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = Y_tr_t[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        train_loss = epoch_loss / max(1, steps)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, Y_val_t).item())
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}  train={train_loss:.6f}  val={val_loss:.6f}')
        best_val = min(best_val, val_loss)

    # Save checkpoint and stats
    ckpt = os.path.join(out_dir, 'checkpoints', 'best_model.pt')
    torch.save({'model_state_dict': model.state_dict(), 'stats': {k: v.squeeze().tolist() for k, v in stats.items()}, 'interior_nodes': interior_nodes.tolist()}, ckpt)
    print('Saved:', ckpt)

    # Plot a few standardized-val examples
    with torch.no_grad():
        pred_val = model(X_val_t).cpu().numpy()
    plot_path = os.path.join(out_dir, 'plots', 'val_samples.png')
    plot_val_examples(X_val_n, Y_val_n, pred_val, plot_path, k=5)
    print('Saved:', plot_path)

    # Loss curve
    loss_plot = os.path.join(out_dir, 'plots', 'loss_curve.png')
    plot_losses(train_losses, val_losses, loss_plot)
    print('Saved:', loss_plot)


if __name__ == '__main__':
    main()


