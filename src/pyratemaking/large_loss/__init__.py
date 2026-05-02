"""Large-loss procedures: basic limits, increased limits factors, layer pricing.

W&M §11. The basic-limits frame caps individual losses at a stated limit so
the data feeding rate-level indication is not dominated by occasional shock
losses. Increased limits factors (ILFs) price the layer above basic limits.
Layer pricing handles arbitrary aggregate excess layers and bridges into
:mod:`actudist` when severity distributions are required.
"""

from pyratemaking.large_loss.basic_limits import basic_limits_losses, cap_at_limit
from pyratemaking.large_loss.increased_limits import (
    increased_limits_factor_table,
    layer_loss_cost,
)
from pyratemaking.large_loss.layer_pricing import layer_pricing_from_distribution

__all__ = [
    "basic_limits_losses",
    "cap_at_limit",
    "increased_limits_factor_table",
    "layer_loss_cost",
    "layer_pricing_from_distribution",
]
