"""RMSE loss function.

Implementation from https://discuss.pytorch.org/t/rmse-loss-function/16540/3.

"""
import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """RMSE loss function module."""

    def __init__(self, eps=1e-6):
        """Initialize loss function."""
        super().__init__()
        self.mse = nn.MSELoss()
        # Add small value to prevent nan in backwards pass
        self.eps = eps

    def forward(self, yhat, y):
        """Forward pass."""
        return torch.sqrt(self.mse(yhat, y) + self.eps)
