# Networks

This repository contains neural network components implemented using Py-Torch.

Each module should have the structure

- `<componentname>.py`: the implementation of the component
- `<componentname.yaml`: default configurations passed to the module when initializing it

The components should extend `torch.nn.Module`. For example:

```python
class NNComponent(nn.Module):
    """Model that implements some NN component"""

    def __init__(self, **kwargs):
        """Initialize the class instance."""
        super().__init__()
        ...
```

Each module should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `utils`

## Network implementations

### `ConvLSTMBlock`

A module implementing a single ConvLSTM layer. The ConvLSTM is introduced in the paper [_Convolutional LSTM network: A machine learning approach for precipitation nowcasting_](https://www.researchwithrutgers.com/en/publications/convolutional-lstm-network-a-machine-learning-approach-for-precip).

### `Encoder`

A module implementing the Encoder model from the article [_Convolutional LSTM network: A machine learning approach for precipitation nowcasting_](https://www.researchwithrutgers.com/en/publications/convolutional-lstm-network-a-machine-learning-approach-for-precip).

### `Decoder`

A module implementing the Decoder model from the article [_Convolutional LSTM network: A machine learning approach for precipitation nowcasting_](https://www.researchwithrutgers.com/en/publications/convolutional-lstm-network-a-machine-learning-approach-for-precip).
