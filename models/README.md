# Models

This repository contains neural networks implemented using PyTorch-Lightning and PyTorch.

Each module should have the structure

- `<modelname>.py`: the implementation of the model
- `<modelname.yaml`: default configurations passed to the module when initializing it

The components should extend `pytorch_lightning.LightningModule`. For example:

```python
class NNModel(pl.LightningModule):
    """Model for the neural network model."""

    def __init__(self, **kwargs):
        """Initialize the class instance."""
        super().__init__()
        ...
```

Each module should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `networks`
- `cost_functions`
- `metrics`
- `utils`

## Network implementations

### `ConvLSTM`

A module implementing the ConvLSTM network described in the paper [_Convolutional LSTM network: A machine learning approach for precipitation nowcasting_](https://www.researchwithrutgers.com/en/publications/convolutional-lstm-network-a-machine-learning-approach-for-precip).
