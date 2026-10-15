"""CNN model mapping a fine mesh-deformation height-field grid to a coarser taxel grid."""

import torch
from torch import nn

from taxelnet.config import ModelConfig


class CNNTaxelPredictor(nn.Module):
    """V0: deterministic spatial CNN, downsampling the mesh grid to the taxel grid.

    The mesh deformation input and taxel output are both regular 2D grids (same kind of
    object, different resolution), so a CNN encoder that exploits 2D locality is a better
    fit than a flatten+MLP (which discards locality) or a GNN (which only earns its keep
    on irregular, unknown connectivity -- not the case for a fixed lattice).

    A stack of conv + pooling blocks progressively downsamples the input regardless of
    its exact resolution (77 x 49 -> 39 x 25 -> 20 x 13 -> 10 x 7 with 3 blocks), then
    AdaptiveAvgPool2d forces the exact target spatial size (7 x 4) so the model doesn't
    need to know the mesh/taxel resolution ratio in advance. Both grids share their
    orientation, so adaptive pooling maps pad regions onto taxels consistently. A linear
    1x1 conv head collapses to a single output channel.

    Extension TODO (V1): a U-Net-style decoder with skip connections and learned
    upsampling, if this plain downsampling CNN underfits spatial detail.
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
        # Linear head (no activation): baseline-corrected taxel values may be negative.
        # It predicts the target in the space set by PreprocessingConfig.target_scaler:
        # - identity: raw tactile counts. The loss is exactly raw-TC L1, but a head
        #   initialized near 0 must climb to ~1,000 TC, so it trains slowly unless the
        #   learning rate or bias initialization is adapted.
        # - global_scale (chosen for V0): raw TC divided by one train-fitted scalar
        #   (target std). L1 stays proportional to raw-TC MAE, so the estimand and the
        #   taxel weighting of L_V0 are unchanged, while outputs are O(1).
        # - per_taxel_zscore: per-taxel standardization. L1 then equals raw L1 weighted
        #   by 1 / sigma_t, so quiet taxels are up-weighted: a different estimand
        #   weighting, to be chosen deliberately.
        # - asinh: y' = asinh(y / s), monotone, so L1 still estimates the raw-TC median
        #   per taxel, but gradients are rebalanced (loud taxels compressed). The inverse
        #   clamps y' to the train range so sinh cannot explode (taxelnet.amp).
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H_mesh, W_mesh) -> returns (B, H_taxel, W_taxel)."""
        features = self.encoder(x)
        resized = self.resize(features)
        out = self.head(resized)
        return out.squeeze(1)


MODELS = {"cnn_v0": CNNTaxelPredictor}


def build_model(model_cfg: ModelConfig, taxel_shape: tuple[int, int]) -> nn.Module:
    """Model registry entry point."""
    if model_cfg.name not in MODELS:
        raise ValueError(
            f"unknown model {model_cfg.name!r}, expected one of {list(MODELS)}"
        )
    return MODELS[model_cfg.name](
        taxel_shape=tuple(taxel_shape),
        base_channels=model_cfg.base_channels,
        num_down_blocks=model_cfg.num_down_blocks,
    )
