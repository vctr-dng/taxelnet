"""CNN model mapping a fine mesh-deformation height-field grid to a coarser taxel grid."""

import torch
from torch import nn


class CNNTaxelPredictor(nn.Module):
    """Downsamples a fine mesh height-field grid to the taxel grid's resolution.

    V1 baseline: the mesh deformation input and taxel output are both regular
    2D grids (same kind of object, different resolution), so a CNN encoder
    that exploits 2D locality is a better fit than a flatten+MLP (which
    discards locality) or a GNN (which only earns its keep on irregular,
    unknown connectivity -- not the case for a fixed lattice).

    A stack of conv + pooling blocks progressively downsamples the input
    regardless of its exact resolution, then AdaptiveAvgPool2d forces the
    exact target spatial size so the model doesn't need to know the
    mesh/taxel resolution ratio in advance. A 1x1 conv collapses to a single
    output channel.

    Extension TODO (not implemented): a fuller U-Net-style decoder with skip
    connections and learned upsampling, if this plain downsampling CNN
    underfits once real data is available.
    """

    def __init__(
        self,
        taxel_shape: tuple[int, int],
        base_channels: int = 16,
        num_down_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.taxel_shape = taxel_shape

        blocks: list[nn.Module] = []
        in_channels = 1
        out_channels = base_channels
        for _ in range(num_down_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                )
            )
            in_channels = out_channels
            out_channels *= 2
        self.encoder = nn.Sequential(*blocks)

        self.resize = nn.AdaptiveAvgPool2d(taxel_shape)
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H_mesh, W_mesh) -> returns (B, H_taxel, W_taxel)."""
        features = self.encoder(x)
        resized = self.resize(features)
        out = self.head(resized)
        return out.squeeze(1)
