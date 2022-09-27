# Datasets

This repository contains custom PyTorch dataset modules.

Each dataset should have the structure

- `<componentname>.py`: the implementation of the component
- `<componentname.yaml`: default configurations passed to the module when initializing it

The datasets should extend `torch.utils.data.Dataset`, for example:

```python
class FMIComposite(Dataset):
    """Dataset for FMI composite in PGM files."""

    def __init__(self, **kwargs):
        """Initialize Dataset."""
        super().__init__()
        ...

    def __len__(self):
        """Mandatory property for Dataset."""
        return self.len

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        ...
        return inputs, outputs, idx
```

Each dataset should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `utils`


## Dataset implementations

### `FMIComposite`

A dataset for FMI radar composites in pgm.gz-format.
