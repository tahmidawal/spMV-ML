#!/usr/bin/env python3
import os
import json
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_fem2d import TwoStageHF2DNet, count_parameters


def timestamp_dir(base: str = 'output') -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    return out_dir

implement the training-loss changes in train_fem2d_npz.py now and leave the optional model tweak for later
def load_npz_and_slice(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    X_tr = z['X_train']
    Y_tr = z['Y_train']
    X_val = z['X_val']
    Y_val = z['Y_val']
    interior_nodes = z.get('interior_nodes', None)
    if interior_nodes is not None:
        X_tr = X_tr[:, interior_nodes]
        Y_tr = Y_tr[:, interior_nodes]
        X_val = X_val[:, interior_nodes]
        Y_val = Y_val[:, interior_nodes]
    return X_tr.astype(np.float32), Y_tr.astype(np.float32), X_val.astype(np.float32), Y_val.astype(np.float32)


def standardize_by_train(X_tr, Y_tr, X_val, Y_val):
    x_mean = X_tr.mean(axis=0, keepdims=True)
    x_std = X_tr.std(axis=0, keepdims=True) + 1e-8
    y_mean = Y_tr.mean(axis=0, keepdims=True)
    y_std = Y_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr_n = (X_tr - x_mean) / x_std
    Y_tr_n = (Y_tr - y_mean) / y_std
    X_val_n = (X_val - x_mean) / x_std
    Y_val_n = (Y_val - y_mean) / y_std
    stats = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
    return X_tr_n, Y_tr_n, X_val_n, Y_val_n, stats


def infer_hw(length: int) -> tuple[int, int]:
    best = (1, length)
    for h in range(1, int(length ** 0.5) + 1):
        if length % h == 0:
            w = length // h
            best = (h, w)
    return best


def reshape_to_2d(X: np.ndarray, H: int, W: int) -> np.ndarray:
    return X.reshape(X.shape[0], 1, H, W)


def plot_val_images(inputs_np, targets_np, preds_np, H, W, save_path):
    import matplotlib.pyplot as plt
    k = min(6, inputs_np.shape[0])
    idx = np.linspace(0, inputs_np.shape[0] - 1, k, dtype=int)
    fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))
    for i, ii in enumerate(idx):
        axes[i, 0].imshow(inputs_np[ii].reshape(H, W), cmap='viridis', aspect='auto')
        axes[i, 0].set_title('input (std)')
        axes[i, 1].imshow(targets_np[ii].reshape(H, W), cmap='viridis', aspect='auto')
        axes[i, 1].set_title('y_true (std)')
        axes[i, 2].imshow(preds_np[ii].reshape(H, W), cmap='viridis', aspect='auto')
        axes[i, 2].set_title('y_pred (std)')
        for j in range(3):
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_val_vectors(inputs_np, targets_np, preds_np, save_path, n=5):
    import matplotlib.pyplot as plt
    k = min(n, inputs_np.shape[0])
    idx = np.linspace(0, inputs_np.shape[0] - 1, k, dtype=int)
    fig, axes = plt.subplots(k, 3, figsize=(14, 2.6 * k))
    if k == 1:
        axes = np.array([axes])
    for i, ii in enumerate(idx):
        x = inputs_np[ii]
        y = targets_np[ii]
        yhat = preds_np[ii]

        # Left: y_true vs y_pred with MSE
        ax0 = axes[i, 0]
        ax0.plot(y, label='y_true (std)', linewidth=0.9)
        ax0.plot(yhat, label='y_pred (std)', linewidth=0.9)
        mse = float(np.mean((y - yhat) ** 2))
        ax0.set_title(f'sample {ii} — y vs yhat, MSE={mse:.3e}')
        ax0.grid(True, alpha=0.25)
        if i == 0:
            ax0.legend(loc='upper right', fontsize=8)

        # Middle: input vector only
        ax1 = axes[i, 1]
        ax1.plot(x, label='x (std)', linewidth=0.9)
        ax1.set_title(f'sample {ii} — input (std)')
        ax1.grid(True, alpha=0.25)
        if i == 0:
            ax1.legend(loc='upper right', fontsize=8)

        # Right: input vs y_true overlay
        ax2 = axes[i, 2]
        ax2.plot(x, label='x (std)', linewidth=0.9)
        ax2.plot(y, label='y_true (std)', linewidth=0.9)
        ax2.set_title(f'sample {ii} — x vs y_true')
        ax2.grid(True, alpha=0.25)
        if i == 0:
            ax2.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
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
    config = {
        'data_path': 'ml_data/fem_sinusoid_k1k2_1to5_300s_20250907_203020.npz',
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 3e-3,
        'weight_decay': 1e-4,
        'trunk_kernel_sizes': [3, 5, 7, 11],
        'trunk_dilations': [1, 2, 4],
        'branch_channels': 4,
        'hf_window': 11,
        'hf_depth': 8,
        'hf_width': 64,
        'gate_channels': 16,
        'posenc_frequencies': [1, 2, 4, 8],
        'lambda_dd': 2e-2,
        'device': 'cpu',
    }

    out_dir = timestamp_dir('output')
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load and standardize on interior nodes
    Xtr, Ytr, Xval, Yval = load_npz_and_slice(config['data_path'])
    Xtr_n, Ytr_n, Xval_n, Yval_n, _ = standardize_by_train(Xtr, Ytr, Xval, Yval)

    # 2D reshape
    L = Xtr_n.shape[1]
    H, W = infer_hw(L)
    Xtr_img = reshape_to_2d(Xtr_n, H, W)
    Ytr_img = reshape_to_2d(Ytr_n, H, W)
    Xval_img = reshape_to_2d(Xval_n, H, W)
    Yval_img = reshape_to_2d(Yval_n, H, W)

    device = torch.device(config['device'])
    model = TwoStageHF2DNet(
        trunk_kernel_sizes=tuple(config['trunk_kernel_sizes']),
        trunk_dilations=tuple(config['trunk_dilations']),
        trunk_branch_channels=config['branch_channels'],
        hf_window=config['hf_window'],
        hf_depth=config['hf_depth'],
        hf_width=config['hf_width'],
        gate_channels=config['gate_channels'],
        posenc_frequencies=tuple(config['posenc_frequencies']),
        activation='snakebeta',
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")

    opt = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()

    # Tensors
    Xtr_t = torch.from_numpy(Xtr_img).to(device)
    Ytr_t = torch.from_numpy(Ytr_img).to(device)
    Xval_t = torch.from_numpy(Xval_img).to(device)
    Yval_t = torch.from_numpy(Yval_img).to(device)

    # Logging
    log_csv = os.path.join(out_dir, 'logs', 'losses.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    train_losses, val_losses = [], []
    epochs = config['epochs']
    bs = config['batch_size']
    Ntr = Xtr_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(Ntr)
        epoch_loss, steps = 0.0, 0
        for i in range(0, Ntr, bs):
            idx = perm[i:i+bs]
            xb = Xtr_t[idx]
            yb = Ytr_t[idx]
            pred = model(xb.squeeze(1))  # (B,H,W) or (B,1,H,W)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            loss = criterion(pred, yb)
            # Match train_fem2d regularization (optional): second-diff consistency
            def lap(t: torch.Tensor) -> torch.Tensor:
                if t.dim() == 3:
                    t = t.unsqueeze(1)
                tpad = nn.functional.pad(t, (1, 1, 1, 1), mode='replicate')
                c = tpad[:, :, 1:-1, 1:-1]
                up = tpad[:, :, :-2, 1:-1]
                down = tpad[:, :, 2:, 1:-1]
                left = tpad[:, :, 1:-1, :-2]
                right = tpad[:, :, 1:-1, 2:]
                return (up + down + left + right - 4 * c)
            loss = loss + config['lambda_dd'] * nn.functional.mse_loss(lap(pred), lap(yb))
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss.item()); steps += 1
        train_loss = epoch_loss / max(1, steps)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            predv = model(Xval_t.squeeze(1))
            if predv.dim() == 3:
                predv = predv.unsqueeze(1)
            vloss = criterion(predv, Yval_t)
            vloss = vloss + config['lambda_dd'] * nn.functional.mse_loss(lap(predv), lap(Yval_t))
            val_loss = float(vloss.item())
            val_losses.append(val_loss)

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss])
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.6f}, val: {val_loss:.6f}")

    # Save checkpoint
    ckpt_path = os.path.join(out_dir, 'checkpoints', 'final_model.pt')
    torch.save({'model_state_dict': model.state_dict(), 'config': config, 'shape': (H, W)}, ckpt_path)
    print('Saved:', ckpt_path)

    # Plots
    with torch.no_grad():
        pred_val = model(Xval_t.squeeze(1))
        if pred_val.dim() == 3:
            pred_val = pred_val.unsqueeze(1)
    preds_np = pred_val.squeeze(1).cpu().numpy().reshape(Xval_n.shape[0], H * W)
    Xval_flat = Xval_n
    Yval_flat = Yval_n
    plot_val_images(Xval_flat, Yval_flat, preds_np, H, W, os.path.join(out_dir, 'plots', 'val_images.png'))
    plot_val_vectors(Xval_flat, Yval_flat, preds_np, os.path.join(out_dir, 'plots', 'val_vectors.png'), n=6)
    plot_losses(train_losses, val_losses, os.path.join(out_dir, 'plots', 'loss_curve.png'))


if __name__ == '__main__':
    main()


