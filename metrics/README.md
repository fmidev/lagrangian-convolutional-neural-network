# Metrics

This repository contains custom TorchMetrics implementations. The metrics should be possible to calculate
for several timesteps in the forecast.

The components should extend `torchmetrics.Metric`. For example:

```python
class CustomMetric(Metric):
    """Custom metric class."""

    def __init__(self, dist_sync_on_step=False, **kwargs):
        """Initilize metric."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        ...

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update calculation states."""
        assert preds.shape == target.shape
        ...

    def compute(self):
        """Return metric value."""
        ...
```

Each module should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `utils`

## Metric implementations

- `CSI`: Critical success index
- `ETS`: Equivalent Threat Score
- `MAE`: Mean absolute error
