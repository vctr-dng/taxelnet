"""CNN with fixed derived input channels (Laplacian, gradient, CoordConv) and capacity knobs.

Capacitive taxels respond to local compression/pressure rather than to top-surface
displacement alone, and their responses are position-specific. `FeatureCNN` therefore
prepends a fixed (non-learned) feature layer that derives extra channels from the
1-channel scaled deformation input, so the deployment contract (B, 1, 77, 49) is unchanged:

- ``z``: the (scaled) deformation itself, as in cnn_v0.
- ``lap``: 5-point discrete Laplacian in 1/mm^2 (node spacing from io_csv rest bounds),
  a curvature / pressure proxy (thin-plate contact pressure ~ bending curvature).
- ``grad``: central-difference gradient magnitude in 1/mm (local slope / shear proxy).
- ``coord``: two CoordConv channels (Liu et al. 2018), row and column in [-1, 1].

Normalisation of ``lap`` and ``grad``: each is divided by a fixed constant so that its
std over the training rows matches the std of the scaled ``z`` channel. The constants
were measured once offline on the *train partition only* of the shape split (split_seed
0, data/newton/v1/training, max_abs input scaling), so no val/test statistics leak.
They are frozen in code (not fitted per run) to keep the model self-contained in
model.pt. ``coord`` is already O(1) and is not rescaled. Derivatives use replicate
padding so the pad border does not create spurious edges.

With the defaults (``input_features="z"``, ``convs_per_block=1``, ``norm="none"``) the
architecture is identical to cnn_v0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from taxelnet.io_csv import SIM_GRID_SHAPE, X_MAX, X_MIN, Y_MAX, Y_MIN

FEATURES = ("z", "lap", "grad", "coord")
NORMS = ("none", "group")

# node spacing (mm): rows run along X, columns along Y (io_csv.SIM_GRID_SHAPE)
DX_ROW = (X_MAX - X_MIN) / (SIM_GRID_SHAPE[0] - 1)
DY_COL = (Y_MAX - Y_MIN) / (SIM_GRID_SHAPE[1] - 1)

# std(channel) / std(z') on the train partition (shape split, seed 0), see module doc
LAP_SCALE = 0.692  # 1/mm^2 (std lap 0.0379 / std z 0.0547, x_scale 2.319 mm)
GRAD_SCALE = 0.335  # 1/mm (std grad 0.0183 / std z 0.0547)


def parse_features(spec: str) -> tuple[str, ...]:
    feats = tuple(f.strip() for f in spec.split(",") if f.strip())
    unknown = [f for f in feats if f not in FEATURES]
    if not feats or unknown or len(set(feats)) != len(feats):
        raise ValueError(f"input_features {spec!r}: expected a comma list of {FEATURES}")
    return feats


class DerivedFeatures(nn.Module):
    """Fixed layer: (B, 1, H, W) scaled deformation -> (B, C, H, W) feature stack."""

    def __init__(
        self,
        features: tuple[str, ...],
        lap_scale: float = LAP_SCALE,
        grad_scale: float = GRAD_SCALE,
    ) -> None:
        super().__init__()
        self.features = features
        self.lap_scale = lap_scale
        self.grad_scale = grad_scale

    @property
    def out_channels(self) -> int:
        return sum(2 if f == "coord" else 1 for f in self.features)

    @staticmethod
    def laplacian(x: torch.Tensor) -> torch.Tensor:
        p = F.pad(x, (1, 1, 1, 1), mode="replicate")
        c = p[..., 1:-1, 1:-1]
        d2r = (p[..., 2:, 1:-1] - 2 * c + p[..., :-2, 1:-1]) / DX_ROW**2
        d2c = (p[..., 1:-1, 2:] - 2 * c + p[..., 1:-1, :-2]) / DY_COL**2
        return d2r + d2c

    @staticmethod
    def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        p = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gr = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / (2 * DX_ROW)
        gc = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / (2 * DY_COL)
        return torch.sqrt(gr * gr + gc * gc + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chans: list[torch.Tensor] = []
        for f in self.features:
            if f == "z":
                chans.append(x)
            elif f == "lap":
                chans.append(self.laplacian(x) / self.lap_scale)
            elif f == "grad":
                chans.append(self.gradient_magnitude(x) / self.grad_scale)
            else:  # coord
                b, _, h, w = x.shape
                rows = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
                cols = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
                rr, cc = torch.meshgrid(rows, cols, indexing="ij")
                chans.append(torch.stack((rr, cc)).expand(b, 2, h, w))
        return chans[0] if len(chans) == 1 else torch.cat(chans, dim=1)


def _norm(kind: str, channels: int) -> list[nn.Module]:
    if kind == "none":
        return []
    groups = 8 if channels % 8 == 0 else 1
    return [nn.GroupNorm(groups, channels)]


class FeatureCNN(nn.Module):
    """cnn_v0 with a derived-feature stem, `convs_per_block` convs and optional GroupNorm."""

    def __init__(
        self,
        taxel_shape: tuple[int, int],
        base_channels: int = 16,
        num_down_blocks: int = 3,
        input_features: str = "z",
        convs_per_block: int = 1,
        norm: str = "none",
    ) -> None:
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm {norm!r}: expected one of {NORMS}")
        if convs_per_block < 1:
            raise ValueError("convs_per_block must be >= 1")
        self.taxel_shape = taxel_shape
        self.features = DerivedFeatures(parse_features(input_features))

        blocks: list[nn.Module] = []
        in_channels = self.features.out_channels
        out_channels = base_channels
        for _ in range(num_down_blocks):
            layers: list[nn.Module] = []
            for _ in range(convs_per_block):
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    *_norm(norm, out_channels),
                    nn.ReLU(inplace=True),
                ]
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, ceil_mode=True))
            blocks.append(nn.Sequential(*layers))
            out_channels *= 2
        self.encoder = nn.Sequential(*blocks)
        self.resize = nn.AdaptiveAvgPool2d(taxel_shape)
        self.head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H_mesh, W_mesh) -> returns (B, H_taxel, W_taxel)."""
        return self.head(self.resize(self.encoder(self.features(x)))).squeeze(1)
