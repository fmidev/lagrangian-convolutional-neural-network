"""Scores from pysteps.verification.detcontscores."""
import numpy as np
from torchmetrics import Metric
import torch


class MAE(Metric):
    """Mean Average Error metric."""

    def __init__(self, length, reduce_dims = (0, 2, 3, 4), dist_sync_on_step=False):
        """Initilize metric."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Length of predictions
        self.length = length
        self.reduce_dims = reduce_dims

        self.add_state(
            "count", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "sum", default=torch.zeros(length), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update calculations."""
        assert preds.shape == target.shape
        preds[~torch.isfinite(preds)] = 0.0
        target[~torch.isfinite(target)] = 0.0

        # compute residuals
        res_ = preds - target

        # Accumulate temp variables
        self.sum += torch.nansum(torch.abs(res_), dim=self.reduce_dims)
        self.count += torch.nansum(torch.isfinite(res_), dim=self.reduce_dims)

    def compute(self):
        """Return metric value."""
        return self.sum / self.count
