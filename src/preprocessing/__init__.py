"""Preprocessing package: normalization stats and train/val/eval splitting."""

from .split import create_split
from .stats import compute_global_stats

__all__ = ["create_split", "compute_global_stats"]
