''' 
Numerically stable Log_cosh loss function (lacking in Pytorch).

Implementation from:
    https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch. 

It works the same way as the Keras implementation:
    https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617
'''

import math

import torch
import torch.nn as nn

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + nn.functional.softplus(-2. * x) - math.log(2.0)
    
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

