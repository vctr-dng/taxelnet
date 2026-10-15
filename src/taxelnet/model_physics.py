"""Physics-structured taxel readout (plan D1) and its CNN hybrid (plan D2).

Footprint model, after Shang et al. (2026) for a 7 x 4 capacitive pad:

    r_t = [ sum_ij w_t(i, j) (K_sigma * P)(i, j) ** eta ] ** (gamma / eta)
    y_t = beta_t r_t + alpha_t sum_ij w_t(i, j) N(i, j) + o_t

with P = relu(x) the indentation, N = relu(-x) the pad bulge (a small linear term),
K_sigma a separable Gaussian spreading kernel (one learnable sigma per axis, in grid
cells), w_t a soft box footprint per taxel (learnable centre, half-widths and edge
softness, normalised to sum 1; initialised to the 7 x 4 block partition of the
77 x 49 grid) and beta_t > 0, alpha_t, o_t per-taxel gain, bulge gain and offset.
For a uniform P the map reduces to beta P ** gamma, so gamma is the load exponent.

All bases are clamped positive (eps) so the fractional powers stay differentiable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from taxelnet.heads import apply_output_head, init_output_head
from taxelnet.io_csv import SIM_GRID_SHAPE

EPS = 1e-4  # in scaled input units (max_abs: max train |x| = 1)


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


class FootprintTaxelPredictor(nn.Module):
    """Spreading -> power eta -> soft-footprint pooling -> power gamma/eta -> gain."""

    def __init__(
        self,
        taxel_shape: tuple[int, int],
        mesh_shape: tuple[int, int] = SIM_GRID_SHAPE,
        kernel_radius: int = 8,
        sigma_init: float = 1.5,
        eta_init: float = 0.7,
        gamma_init: float = 0.5,
        gain_init: float = 8.0,
        taxel_affine_head: bool = False,
        head_activation: str = "linear",
        head_lower: float | None = None,
        head_upper: float | None = None,
        **_: object,  # base_channels / num_down_blocks from ModelConfig: unused
    ) -> None:
        super().__init__()
        self.taxel_shape = tuple(taxel_shape)
        self.mesh_shape = tuple(mesh_shape)
        th, tw = self.taxel_shape
        mh, mw = self.mesh_shape
        n_taxels = th * tw

        self.kernel_radius = kernel_radius
        self.log_sigma = nn.Parameter(torch.full((2,), math.log(sigma_init)))
        self.eta_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_init)))
        self.gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(gamma_init)))

        # soft boxes, in grid-index units, initialised to the block partition
        bh, bw = mh / th, mw / tw
        r, c = torch.meshgrid(torch.arange(th), torch.arange(tw), indexing="ij")
        self.centre = nn.Parameter(
            torch.stack(
                [(r.flatten() + 0.5) * bh - 0.5, (c.flatten() + 0.5) * bw - 0.5], 1
            ).float()
        )
        self.log_half = nn.Parameter(
            torch.log(torch.tensor([bh / 2, bw / 2])).repeat(n_taxels, 1)
        )
        self.log_tau = nn.Parameter(torch.full((n_taxels,), math.log(0.5)))
        self.register_buffer("rows", torch.arange(mh).float(), persistent=False)
        self.register_buffer("cols", torch.arange(mw).float(), persistent=False)

        self.log_gain = nn.Parameter(torch.full((n_taxels,), math.log(gain_init)))
        self.bulge_gain = nn.Parameter(torch.zeros(n_taxels))
        self.offset = nn.Parameter(torch.zeros(n_taxels))
        # optional output heads on the final map (taxelnet.heads)
        init_output_head(
            self, self.taxel_shape, taxel_affine_head, head_activation,
            head_lower, head_upper,
        )  # fmt: skip

    # -- physical parameters ------------------------------------------------------
    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.eta_raw).clamp(0.05, 3.0)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw).clamp(0.05, 3.0)

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(0.05, float(self.kernel_radius))

    def footprint_weights(self) -> torch.Tensor:
        """(T, H, W) soft footprints, each summing to 1."""
        half = self.log_half.exp()
        tau = self.log_tau.exp().clamp(min=0.05)[:, None]
        du = self.rows[None, :] - self.centre[:, :1]  # (T, H)
        dv = self.cols[None, :] - self.centre[:, 1:]  # (T, W)
        wu = torch.sigmoid((half[:, :1] - du.abs()) / tau)
        wv = torch.sigmoid((half[:, 1:] - dv.abs()) / tau)
        w = wu[:, :, None] * wv[:, None, :]
        return w / w.sum((1, 2), keepdim=True).clamp(min=1e-8)

    def _spread(self, p: torch.Tensor) -> torch.Tensor:
        k = torch.arange(-self.kernel_radius, self.kernel_radius + 1, device=p.device)
        g = torch.exp(-0.5 * (k[None, :].float() / self.sigma[:, None]) ** 2)
        g = g / g.sum(1, keepdim=True)  # (2, K)
        p = F.conv2d(p, g[0].view(1, 1, -1, 1), padding=(self.kernel_radius, 0))
        return F.conv2d(p, g[1].view(1, 1, 1, -1), padding=(0, self.kernel_radius))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H_mesh, W_mesh) -> (B, H_taxel, W_taxel)."""
        w = self.footprint_weights()  # (T, H, W)
        spread = self._spread(x.clamp(min=0)).clamp(min=0)[:, 0]  # (B, H, W)
        eta, gamma = self.eta, self.gamma
        powered = (spread + EPS) ** eta
        pooled = torch.einsum("bhw,thw->bt", powered, w).clamp(min=EPS**eta)
        r = pooled ** (gamma / eta)
        bulge = torch.einsum("bhw,thw->bt", (-x).clamp(min=0)[:, 0], w)
        y = self.log_gain.exp() * r + self.bulge_gain * bulge + self.offset
        return apply_output_head(self, y.view(-1, *self.taxel_shape))

    # -- reporting ----------------------------------------------------------------
    @torch.no_grad()
    def physical_params(self) -> dict[str, float]:
        gain = self.log_gain.exp()
        half = self.log_half.exp()
        return {
            "eta": self.eta.item(),
            "gamma": self.gamma.item(),
            "sigma_rows": self.sigma[0].item(),
            "sigma_cols": self.sigma[1].item(),
            "gain_min": gain.min().item(),
            "gain_median": gain.median().item(),
            "gain_max": gain.max().item(),
            "offset_mean": self.offset.mean().item(),
            "bulge_gain_mean": self.bulge_gain.mean().item(),
            "half_rows_mean": half[:, 0].mean().item(),
            "half_cols_mean": half[:, 1].mean().item(),
            "tau_mean": self.log_tau.exp().mean().item(),
        }

    @torch.no_grad()
    def report(self, out_dir: Path) -> dict[str, float]:
        """Save footprint_weights.png + physical_params.json; return scalar params."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        params = self.physical_params()
        w = self.footprint_weights().cpu().numpy()
        th, tw = self.taxel_shape
        fig, axes = plt.subplots(th, tw, figsize=(2.0 * tw, 2.2 * th))
        for t, ax in enumerate(axes.flat):
            ax.imshow(w[t], cmap="viridis", aspect="auto")
            ax.set_title(f"tax{t + 1}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            "learned footprints  eta={eta:.2f} gamma={gamma:.2f} "
            "sigma=({sigma_rows:.2f},{sigma_cols:.2f}) cells".format(**params),
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "footprint_weights.png", dpi=110)
        plt.close(fig)
        extra = {
            "gain": self.log_gain.exp().tolist(),
            "offset": self.offset.tolist(),
            "bulge_gain": self.bulge_gain.tolist(),
            "centre": self.centre.tolist(),
            "half": self.log_half.exp().tolist(),
        }
        (out_dir / "physical_params.json").write_text(
            json.dumps({**params, **extra}, indent=2)
        )
        return params


class FootprintCNNTaxelPredictor(nn.Module):
    """y = footprint(x) + cnn_v0(x); the CNN residual is kept in `self.residual` so the
    training loop can penalise it (train.residual_l1_weight)."""

    def __init__(
        self,
        taxel_shape: tuple[int, int],
        base_channels: int = 16,
        num_down_blocks: int = 3,
        taxel_affine_head: bool = False,
        head_activation: str = "linear",
        head_lower: float | None = None,
        head_upper: float | None = None,
        **kwargs: object,
    ) -> None:
        from taxelnet.model import CNNTaxelPredictor  # lazy: model.py registers us

        super().__init__()
        self.footprint = FootprintTaxelPredictor(taxel_shape, **kwargs)
        self.cnn = CNNTaxelPredictor(taxel_shape, base_channels, num_down_blocks)
        # start from the pure physics model
        nn.init.zeros_(self.cnn.head.weight)
        nn.init.zeros_(self.cnn.head.bias)
        self.residual: torch.Tensor | None = None
        # optional output heads on the sum footprint + residual (taxelnet.heads); the
        # two submodules keep plain linear outputs
        init_output_head(
            self, tuple(taxel_shape), taxel_affine_head, head_activation,
            head_lower, head_upper,
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.residual = self.cnn(x)
        return apply_output_head(self, self.footprint(x) + self.residual)

    def report(self, out_dir: Path) -> dict[str, float]:
        return self.footprint.report(out_dir)
