"""Optional output heads shared by every registered model (lab/p2 integration).

Applied to the model output in the scaled target space, in this order:

1. `taxel_affine_head` (lab/calib, plan C1): y_t = g_t * z_t + o_t, init identity;
   parameters `taxel_gain` / `taxel_offset`.
2. `head_activation = "tanh_bounded"` (lab/sat, plan C2): asymmetric soft clip to
   (lo, hi); module `out_act` (buffers `lo` / `hi`). Last, so the bound always holds.

Both are attached directly to the top-level module under these names, so cnn_v0
state_dicts from the calib / sat branches load unchanged, and default models carry no
extra parameters. For hybrid models (footprint_cnn) they act on the final sum
footprint + residual, not on the CNN residual alone.
"""

from __future__ import annotations

import torch
from torch import nn

from taxelnet.saturation import HEAD_ACTIVATIONS, BoundedTanh


def init_output_head(
    module: nn.Module,
    taxel_shape: tuple[int, int],
    taxel_affine_head: bool = False,
    head_activation: str = "linear",
    head_lower: float | None = None,
    head_upper: float | None = None,
) -> None:
    module.taxel_affine_head = taxel_affine_head
    if taxel_affine_head:
        module.taxel_gain = nn.Parameter(torch.ones(taxel_shape))
        module.taxel_offset = nn.Parameter(torch.zeros(taxel_shape))
    if head_activation not in HEAD_ACTIVATIONS:
        raise ValueError(f"head_activation must be one of {HEAD_ACTIVATIONS}")
    # absent for "linear" so the default state_dict is unchanged
    if head_activation == "tanh_bounded":
        module.out_act = BoundedTanh(
            -1.0 if head_lower is None else head_lower,
            1.0 if head_upper is None else head_upper,
        )


def apply_output_head(module: nn.Module, out: torch.Tensor) -> torch.Tensor:
    """out: (B, H_taxel, W_taxel) in scaled target units."""
    if getattr(module, "taxel_affine_head", False):
        out = out * module.taxel_gain + module.taxel_offset
    if hasattr(module, "out_act"):
        out = module.out_act(out)
    return out
