import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class SnakeBeta(nn.Module):
    """Snake activation with learned beta (periodic; suitable for HF)."""

    def __init__(self, beta_init: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.beta, min=1e-3)
        return x + torch.sin(beta * x) ** 2 / beta


class LowFreqTrunk2D(nn.Module):
    """Linear multi-branch Conv2d trunk capturing low-frequency structure.

    Returns y_low (B,1,H,W) and concatenated features F (B,K,H,W).
    """

    def __init__(self,
                 kernel_sizes=(3, 5, 7, 11),
                 dilations=(1, 2, 4),
                 branch_channels: int = 4,
                 bias: bool = False):
        super().__init__()
        branches = []
        for k in kernel_sizes:
            for d in dilations:
                padding = (k // 2) * d
                branches.append(
                    nn.Conv2d(1, branch_channels, kernel_size=k, dilation=d, padding=padding, bias=bias)
                )
        self.branches = nn.ModuleList(branches)
        self.out = nn.Conv2d(len(branches) * branch_channels, 1, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,H,W)
        feats = [b(x) for b in self.branches]  # list of (B,C,H,W)
        Fcat = torch.cat(feats, dim=1)  # (B,K,H,W)
        y_low = self.out(Fcat)  # (B,1,H,W)
        return y_low, Fcat


class SpikeGate2D(nn.Module):
    """Position-conditioned gate g(h,w) in [0,1] to focus HF residual."""

    def __init__(self, in_channels: int = 1, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, xin: torch.Tensor) -> torch.Tensor:
        if xin.dim() == 3:
            xin = xin.unsqueeze(1)
        return self.net(xin)  # (B,1,H,W)


class HFResidualHeadSiren2D(nn.Module):
    """High-frequency residual from local 2D patches via deep Snake/SIREN MLP.

    Per-pixel vector: [patch_flat, Fmean(h,w), PE(h,w)] → scalar residual r(h,w).
    """

    def __init__(self,
                 patch_window: int = 11,
                 mlp_depth: int = 8,
                 mlp_width: int = 64,
                 activation: str = 'snakebeta',
                 posenc_frequencies: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        assert patch_window % 2 == 1 and patch_window >= 3
        self.w = patch_window
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.activation_name = activation
        self.posenc_frequencies = tuple(posenc_frequencies)
        self.mlp = None  # lazily built with correct input dim

    def _build_mlp(self, in_dim: int):
        layers = []
        act = SnakeBeta() if self.activation_name == 'snakebeta' else nn.Tanh()
        layers.append(nn.Linear(in_dim, self.mlp_width))
        layers.append(act)
        for _ in range(self.mlp_depth - 2):
            layers.append(nn.Linear(self.mlp_width, self.mlp_width))
            layers.append(act)
        layers.append(nn.Linear(self.mlp_width, 1))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, Fcat: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W) or (B,H,W); Fcat: (B,K,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, _, H, W = x.shape

        # Unfold local patches
        patches = F.unfold(x, kernel_size=self.w, padding=self.w // 2)  # (B, w*w, H*W)
        patches = patches.transpose(1, 2)  # (B, H*W, w*w)

        # Mean feature per pixel
        Fmean = Fcat.mean(dim=1, keepdim=True)  # (B,1,H,W)
        Fmean_flat = Fmean.flatten(2).transpose(1, 2)  # (B, H*W, 1)

        # Positional encodings per pixel
        ys = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
        pe_list = [yy, xx]
        for f in self.posenc_frequencies:
            angy = 2 * math.pi * f * yy
            angx = 2 * math.pi * f * xx
            pe_list.append(torch.sin(angy))
            pe_list.append(torch.cos(angy))
            pe_list.append(torch.sin(angx))
            pe_list.append(torch.cos(angx))
        pe = torch.stack(pe_list, dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,Cpe,H,W)
        pe_flat = pe.flatten(2).transpose(1, 2)  # (B, H*W, Cpe)

        # Build MLP if needed
        in_dim = patches.shape[-1] + 1 + pe_flat.shape[-1]
        if self.mlp is None:
            self._build_mlp(in_dim)

        # Assemble and run MLP
        vec = torch.cat([patches, Fmean_flat, pe_flat], dim=-1)  # (B, H*W, in_dim)
        vec = vec.reshape(B * H * W, in_dim)
        out = self.mlp(vec).reshape(B, H, W, 1).permute(0, 3, 1, 2)  # (B,1,H,W)
        return out


class TwoStageHF2DNet(nn.Module):
    """2D two-stage: low-freq trunk + gated high-freq residual."""

    def __init__(self,
                 trunk_kernel_sizes=(3, 5, 7, 11),
                 trunk_dilations=(1, 2, 4),
                 trunk_branch_channels: int = 4,
                 hf_window: int = 11,
                 hf_depth: int = 8,
                 hf_width: int = 64,
                 gate_channels: int = 16,
                 posenc_frequencies: tuple[int, ...] = (1, 2, 4, 8),
                 activation: str = 'snakebeta'):
        super().__init__()
        self.trunk = LowFreqTrunk2D(trunk_kernel_sizes, trunk_dilations, trunk_branch_channels, bias=False)
        self.gate = SpikeGate2D(in_channels=1, hidden=gate_channels)
        self.hf = HFResidualHeadSiren2D(
            patch_window=hf_window,
            mlp_depth=hf_depth,
            mlp_width=hf_width,
            activation=activation,
            posenc_frequencies=posenc_frequencies,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_low, Fcat = self.trunk(x)     # (B,1,H,W), (B,K,H,W)
        gate = self.gate(x)             # (B,1,H,W)
        if x.dim() == 3:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
        r = self.hf(x_in, Fcat)         # (B,1,H,W)
        return (y_low + gate * r).squeeze(1)


