"""Scores from pysteps.verification.detcatscores."""
from torchmetrics import Metric
import torch


class CSI(Metric):
    """Critical Success Index metric."""

    def __init__(self, threshold, length, reduce_dims = (0, 2, 3, 4), dist_sync_on_step=False):
        """Initilize metric."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Threshold applied to data
        self.threshold = threshold
        # Length of predictions
        self.length = length
        self.reduce_dims = reduce_dims

        self.add_state(
            "hits", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "false_alarms", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "misses", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "correct_negatives", default=torch.zeros(length), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update calculations."""
        assert preds.shape == target.shape
        preds[~torch.isfinite(preds)] = 0.0
        target[~torch.isfinite(target)] = 0.0

        predb = preds > self.threshold
        obsb = target > self.threshold

        # calculate hits, misses, false positives, correct rejects
        H_idx = torch.logical_and(predb == 1, obsb == 1)
        F_idx = torch.logical_and(predb == 1, obsb == 0)
        M_idx = torch.logical_and(predb == 0, obsb == 1)
        R_idx = torch.logical_and(predb == 0, obsb == 0)

        # accumulate in the contingency table
        self.hits += torch.nansum(H_idx.int(), dim=self.reduce_dims)
        self.misses += torch.nansum(M_idx.int(), dim=self.reduce_dims)
        self.false_alarms += torch.nansum(F_idx.int(), dim=self.reduce_dims)
        self.correct_negatives += torch.nansum(R_idx.int(), dim=self.reduce_dims)

    def compute(self):
        """Return metric value."""
        H = 1.0 * self.hits  # true positives
        M = 1.0 * self.misses  # false negatives
        F = 1.0 * self.false_alarms  # false positives
        # R = 1.0 * self.correct_negatives  # true negatives

        # POD = H / (H + M)
        # FAR = F / (H + F)
        # FA = F / (F + R)
        # s = (H + M) / (H + M + F + R)
        CSI = H / (H + M + F)
        
        CSI[~torch.isfinite(CSI)] = 0.0
        return CSI


class ETS(Metric):
    """Equivalent Threat Score metric."""

    def __init__(self, threshold, length, reduce_dims = (0, 2, 3, 4), dist_sync_on_step=False):
        """Initilize metric."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Threshold applied to data
        self.threshold = threshold
        # Length of predictions
        self.length = length
        self.reduce_dims = reduce_dims

        self.add_state(
            "hits", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "false_alarms", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "misses", default=torch.zeros(length), dist_reduce_fx="sum")
        self.add_state(
            "correct_negatives", default=torch.zeros(length), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update calculations."""
        assert preds.shape == target.shape
        preds[~torch.isfinite(preds)] = 0.0
        target[~torch.isfinite(target)] = 0.0
        
        predb = preds > self.threshold
        obsb = target > self.threshold

        # calculate hits, misses, false positives, correct rejects
        H_idx = torch.logical_and(predb == 1, obsb == 1)
        F_idx = torch.logical_and(predb == 1, obsb == 0)
        M_idx = torch.logical_and(predb == 0, obsb == 1)
        R_idx = torch.logical_and(predb == 0, obsb == 0)

        # accumulate in the contingency table
        self.hits += torch.nansum(H_idx.int(), dim=self.reduce_dims)
        self.misses += torch.nansum(M_idx.int(), dim=self.reduce_dims)
        self.false_alarms += torch.nansum(F_idx.int(), dim=self.reduce_dims)
        self.correct_negatives += torch.nansum(R_idx.int(), dim=self.reduce_dims)

    def compute(self):
        """Return metric value."""
        H = 1.0 * self.hits  # true positives
        M = 1.0 * self.misses  # false negatives
        F = 1.0 * self.false_alarms  # false positives
        R = 1.0 * self.correct_negatives  # true negatives

        POD = H / (H + M)
        # FAR = F / (H + F)
        FA = F / (F + R)
        s = (H + M) / (H + M + F + R)
        ETS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
        
        ETS[~torch.isfinite(ETS)] = 0.0

        return ETS
