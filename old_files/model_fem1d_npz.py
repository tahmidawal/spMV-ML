import torch
import torch.nn as nn


class LocalSpMV1DNet(nn.Module):
    """Local 1D convolution network approximating a 1D sparse operator.

    Copied from model_fem1d.LocalSpMV1DNet with defaults suitable for linear FEM mapping.
    Uses linear layers (no activations by default) to respect operator linearity.
    """

    def __init__(
        self,
        kernel_sizes=(1, 3, 5, 7, 11, 15),
        dilations=(1, 2, 4, 8),
        branch_channels: int = 8,
        activation: str = 'identity',  # 'identity' | 'tanh' | 'gelu'
        bias: bool = False,
        refiner_layers: int = 1,
        refiner_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        branches: list[nn.Module] = []
        for kernel_size in kernel_sizes:
            for dilation in dilations:
                padding = (kernel_size // 2) * dilation
                conv = nn.Conv1d(
                    in_channels=1,
                    out_channels=branch_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                )
                branches.append(conv)
        self.branches = nn.ModuleList(branches)

        num_branches = len(kernel_sizes) * len(dilations)
        self.combiner = nn.Conv1d(
            in_channels=branch_channels * num_branches,
            out_channels=1,
            kernel_size=1,
            bias=bias,
        )

        if activation == 'identity':
            self.activation: nn.Module | None = None
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.refiners = nn.ModuleList()
        if refiner_layers > 0:
            pad = (refiner_kernel_size // 2)
            for _ in range(refiner_layers):
                self.refiners.append(
                    nn.Conv1d(1, 1, kernel_size=refiner_kernel_size, padding=pad, bias=bias)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,L)

        features = [branch(x) for branch in self.branches]
        concatenated = torch.cat(features, dim=1)
        y = self.combiner(concatenated)

        if self.activation is not None:
            y = self.activation(y)

        for conv in self.refiners:
            y = conv(y)
            if self.activation is not None:
                y = self.activation(y)

        return y.squeeze(1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


